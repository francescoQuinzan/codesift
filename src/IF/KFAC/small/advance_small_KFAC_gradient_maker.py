import re
from collections import defaultdict
from typing import Dict, Optional, Tuple, Iterable

import torch
from torch import nn
import torch.distributed as dist

from asdl.precondition import (
    PreconditioningConfig,
    KfacGradientMaker,
    NaturalGradientMaker,
)
from asdl.fisher import FisherConfig, get_fisher_maker, LOSS_CROSS_ENTROPY
from asdl.matrices import FISHER_EMP, SHAPE_KRON
from asdl.operations.linear import Linear as OpLinear
from asdl.operations.conv import Conv2d as OpConv2d


from asdl.matrices import *
from asdl.symmatrix import *

from IF.KFAC.matrix_operations import (
    _chol_inv,
    _chol_with_adaptive_damping,
    _mean_eig_scale,
    _precond_down_full,
    _precond_up_full,
    _NAME_PAT_MLP,
    _NAME_PAT_Q,
    _NAME_PAT_K,
    _NAME_PAT_V,
    _INVALID_EMA_DECAY,
)

from pathlib import Path


class AdvancedSmallKfacGradientMaker(NaturalGradientMaker):
    """Configurable Small-KFAC variant with optional Fisher stats and KFAC sync.

    # Extensions to AdvancedSmallKFAC:
    # - proj_names=None: precondition all eligible layers
    # - small_matrices=True: store activations (A) and gradients (G) instead of full matrices
    # - short_side=True: use short-side factor only; False: use both sides
    # - use_woodbury=True: apply Woodbury approximation on long side
    #   (only valid when short_side=True; otherwise ignored)
    # - detect_device: auto-detect devices and shard curvature matrices across GPUs,
    #   enabling multi-GPU storage without torchrun/DDP

    """

    def __init__(
        self,
        model: nn.Module,
        config: PreconditioningConfig,
        *,
        rel_eps: float = 0.1,
        normalize: bool = True,
        use_amp: bool = True,
        factor_device: Optional[torch.device | str] = None,
        use_grad_gram: bool = True,
        use_woodbury: bool = True,
        use_kfac_sync: bool = False,
        use_kfac_preconditioner_update: bool = False,
        use_kfac_curvature_update: bool = False,
        use_small_cache: bool = False,
        fisher_type: str = FISHER_EMP,
        loss_type: str = LOSS_CROSS_ENTROPY,
        scale: float = 1.0,
        seed: Optional[int] = None,
        fisher_attr: str = "fisher_small",
        proj_names: Optional[Tuple[str, ...]] = None,
        name_pattern: Optional[str] = "default",
        small_matrices: bool = False,
        short_side: bool = True,
        detect_device: Optional[Tuple[torch.device | str, ...] | bool] = True,
        exclude_cuda0: bool = False,
        kfac_sync_kwargs: Optional[Dict[str, object]] = None,
        kfac_preconditioner_kwargs: Optional[Dict[str, object]] = None,
        min_damping: float = 1e-3,
        save_dir: Optional[Path] = None,
    ):
        # Usage:
        # gm = AdvancedSmallKfacGradientMaker(model, config, proj_names=("up_proj", "down_proj", "q_proj"))
        # gm.set_target_layers([0, 1, 3])
        # gm.set_target_projs(["q_proj", "k_proj"])
        super().__init__(model, config)
        self.rel_eps = rel_eps
        self.normalize = normalize
        self.use_amp = use_amp
        self.factor_device = (
            torch.device(factor_device) if factor_device is not None else None
        )
        self.min_damping = min_damping
        self.A_u_dict = defaultdict(lambda: None)
        self.G_d_dict = defaultdict(lambda: None)
        self.counts = defaultdict(int)
        self.use_grad_gram = use_grad_gram
        self.short_side = short_side
        self.use_woodbury = use_woodbury if short_side else None
        self.use_kfac_preconditioner_update = use_kfac_preconditioner_update
        self.use_kfac_curvature_update = use_kfac_curvature_update
        self.use_small_cache = use_small_cache
        self.scale = scale
        self.small_matrices = small_matrices
        self._target_layers = None
        if proj_names is None:
            self._proj_names = {"up_proj", "down_proj"}
        else:
            self._proj_names = {str(p) for p in proj_names}
        if name_pattern is None:
            self._name_pattern = None
        elif name_pattern == "default":
            self._name_pattern = "default"
        else:
            self._name_pattern = re.compile(str(name_pattern))

        self._factor_devices = self._resolve_factor_devices(
            detect_device, factor_device, exclude_cuda0
        )
        self._device_map = {}
        self._device_rr = 0

        self.A_qkv_dict = defaultdict(lambda: None)
        self.G_qkv_dict = defaultdict(lambda: None)
        self.qkv_counts = defaultdict(int)

        self._param_A = defaultdict(lambda: None)
        self._param_G = defaultdict(lambda: None)
        self._param_counts = defaultdict(int)
        self._raw_inputs = {}
        self._raw_outgrads = {}
        self._io_hooks = []
        self._param_to_module = self._build_param_module_map()
        print("AdvancedSmallKfacGradientMaker target params:")

        for name, _ in sorted(self._iter_target_params(), key=lambda item: item[0]):
            print(f"  - {name}")
        print(
            "AdvancedSmallKfacGradientMaker planned curvature path:",
            self._describe_curvature_path(),
        )

        self.fisher_maker = None
        if not use_grad_gram:
            fisher_config = FisherConfig(
                fisher_type=fisher_type,
                fisher_shapes=[SHAPE_KRON],
                loss_type=loss_type,
                seed=seed,
                ignore_modules=config.ignore_modules,
                fisher_attr=fisher_attr,
            )
            self.fisher_maker = get_fisher_maker(model, fisher_config)

        self._kfac_syncer = None
        if use_kfac_sync or use_kfac_curvature_update:
            self._init_kfac_syncer(kfac_sync_kwargs)
        self._kfac_preconditioner_kwargs = (
            {}
            if kfac_preconditioner_kwargs is None
            else dict(kfac_preconditioner_kwargs)
        )

        self._cache_mlp = {}
        self._cache_qkv = {}

        hashed_args = abs(
            hash(
                (
                    model,
                    rel_eps,
                    normalize,
                    use_amp,
                    factor_device,
                    use_grad_gram,
                    use_woodbury,
                    use_kfac_sync,
                    use_kfac_preconditioner_update,
                    use_kfac_curvature_update,
                    use_small_cache,
                    fisher_type,
                    loss_type,
                    scale,
                    seed,
                    fisher_attr,
                    proj_names,
                    name_pattern,
                    small_matrices,
                    short_side,
                )
            )
        )
        self.save_state_path = (
            save_dir / f"{hashed_args}.pt" if save_dir is not None else None
        )

    def get_model_layer_indices(self):
        indices = set()
        for name, _ in self.model.named_parameters():
            match = (
                _NAME_PAT_MLP.match(name)
                or _NAME_PAT_Q.match(name)
                or _NAME_PAT_K.match(name)
                or _NAME_PAT_V.match(name)
            )
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def get_model_layer_count(self) -> int:
        return len(self.get_model_layer_indices())

    def set_target_layers(self, layers):
        if layers is None:
            self._target_layers = None
            return
        self._target_layers = {int(layer) for layer in layers}

    def get_target_layers(self):
        if self._target_layers is None:
            return None
        return sorted(self._target_layers)

    def set_target_projs(self, projs):
        if projs is None:
            self._proj_names = {"up_proj", "down_proj"}
            return
        self._proj_names = {str(p) for p in projs}

    def get_target_projs(self):
        return sorted(self._proj_names)

    def do_forward_and_backward(self, step=None) -> bool:
        return not self.do_update_curvature(step)

    def _resolve_factor_devices(
        self, detect_device, factor_device, exclude_cuda0=False
    ):
        if factor_device is not None:
            return [torch.device(factor_device)]
        if detect_device is True:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                devices = [
                    torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
                ]
                if exclude_cuda0:
                    devices = [
                        d for d in devices if not (d.type == "cuda" and d.index == 0)
                    ]
                return devices or [torch.device("cpu")]
            return [torch.device("cpu")]
        if isinstance(detect_device, (list, tuple)):
            devices = [torch.device(d) for d in detect_device]
            if exclude_cuda0:
                devices = [
                    d for d in devices if not (d.type == "cuda" and d.index == 0)
                ]
            return devices
        return []

    def _device_for_key(self, key, fallback):
        if not self._factor_devices:
            return fallback
        if key not in self._device_map:
            device = self._factor_devices[self._device_rr % len(self._factor_devices)]
            self._device_rr += 1
            self._device_map[key] = device
        return self._device_map[key]

    def _build_param_module_map(self):
        mapping = {}
        ignore_modules = self.config.ignore_modules
        for module_name, module in self.model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if not module.weight.requires_grad:
                continue
            if ignore_modules is not None:
                ignored = False
                for ignore_module in ignore_modules:
                    if isinstance(ignore_module, type):
                        if isinstance(module, ignore_module):
                            ignored = True
                            break
                    elif isinstance(ignore_module, str):
                        if ignore_module in module_name:
                            ignored = True
                            break
                    elif ignore_module is module:
                        ignored = True
                        break
                if ignored:
                    continue
            key = f"{module_name}.weight" if module_name else "weight"
            mapping[key] = module
        return mapping

    def _is_ignored_module(self, module_name: str, module: nn.Module) -> bool:
        ignore_modules = self.config.ignore_modules
        if not ignore_modules:
            return False
        for ignore_module in ignore_modules:
            if isinstance(ignore_module, type):
                if isinstance(module, ignore_module):
                    return True
            elif isinstance(ignore_module, str):
                if ignore_module in module_name:
                    return True
            elif ignore_module is module:
                return True
        return False

    def _is_ignored_param(self, name: str) -> bool:
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            return False
        return self._is_ignored_module(module_name, module)

    def _describe_curvature_path(self) -> str:
        if self.small_matrices:
            return "generic/small_matrices"
        if (
            self._name_pattern is None
            or isinstance(self._name_pattern, re.Pattern)
            or not self.short_side
        ):
            return "generic/grad_gram"
        if self.use_kfac_curvature_update:
            return "kfac_syncer"
        if self.use_grad_gram:
            return "small_grad_gram"
        return "fisher_fallback"

    def _use_generic_precond(self):
        return (
            self.small_matrices
            or self._name_pattern is None
            or isinstance(self._name_pattern, re.Pattern)
            or not self.short_side
        )

    def _match_default_param(self, name):
        match = (
            _NAME_PAT_MLP.match(name)
            or _NAME_PAT_Q.match(name)
            or _NAME_PAT_K.match(name)
            or _NAME_PAT_V.match(name)
        )
        if not match:
            return None
        layer = int(match.group(1))
        proj = match.group(2)
        if self._target_layers is not None and layer not in self._target_layers:
            return None
        if proj not in self._proj_names:
            return None
        return layer, proj

    def _iter_target_params(self) -> Iterable[Tuple[str, nn.Parameter]]:
        if self._name_pattern is None:
            for name, p in self.model.named_parameters():
                if not p.requires_grad or p.ndim not in (2, 4):
                    continue
                if self._is_ignored_param(name):
                    continue
                yield name, p
            return
        if isinstance(self._name_pattern, re.Pattern):
            for name, p in self.model.named_parameters():
                if not p.requires_grad or p.ndim not in (2, 4):
                    continue
                if self._is_ignored_param(name):
                    continue
                if self._name_pattern.match(name):
                    yield name, p
            return
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.ndim not in (2, 4):
                continue
            if self._is_ignored_param(name):
                continue
            if self._match_default_param(name) is None:
                continue
            yield name, p

    def _install_io_hooks(self):
        self._raw_inputs = {}
        self._raw_outgrads = {}
        self._io_hooks = []
        target_names = {name for name, _ in self._iter_target_params()}
        for name, module in self._param_to_module.items():
            if module.weight.ndim not in (2, 4):
                continue
            if name not in target_names:
                continue

            def _fwd_hook(mod, inputs, outputs, key=name):
                if not inputs:
                    return
                self._raw_inputs[key] = inputs[0].detach()

            def _bwd_hook(mod, grad_inputs, grad_outputs, key=name):
                if not grad_outputs:
                    return
                self._raw_outgrads[key] = grad_outputs[0].detach()

            self._io_hooks.append(module.register_forward_hook(_fwd_hook))
            self._io_hooks.append(module.register_full_backward_hook(_bwd_hook))

    def _clear_io_hooks(self):
        for handle in self._io_hooks:
            handle.remove()
        self._io_hooks = []

    def _compute_cov_from_raw(self, name, device):
        module = self._param_to_module.get(name)
        if module is None:
            return None, None
        in_data = self._raw_inputs.get(name)
        out_grads = self._raw_outgrads.get(name)
        if in_data is None or out_grads is None:
            return None, None
        if isinstance(module, nn.Linear):
            op = OpLinear
        elif isinstance(module, nn.Conv2d):
            op = OpConv2d
        else:
            return None, None
        in_data = op.preprocess_in_data(module, in_data, None).to(device)
        out_grads = op.preprocess_out_grads(module, out_grads).to(device)
        A = op.cov_kron_A(module, in_data)
        G = op.cov_kron_B(module, out_grads)
        return A, G

    def _accumulate_param_factors_from_grads(self):
        ema_decay = self.config.ema_decay
        use_ema = ema_decay != _INVALID_EMA_DECAY
        new_A = defaultdict(lambda: None)
        new_G = defaultdict(lambda: None)
        new_counts = defaultdict(int)

        for name, p in self._iter_target_params():
            if p.grad is None:
                continue
            gW = p.grad.detach()
            if self.normalize:
                nd = gW.norm()
                if nd > 0:
                    gW = gW / nd
            if gW.ndim == 4:
                gW = gW.flatten(start_dim=1)
            elif gW.ndim != 2:
                continue
            factor_device = self._device_for_key(name, gW.device)
            gW = gW.to(factor_device)
            A_i = gW.T @ gW
            G_i = gW @ gW.T
            if new_A[name] is None:
                new_A[name] = torch.zeros_like(
                    A_i, device=factor_device, dtype=torch.float32
                )
            if new_G[name] is None:
                new_G[name] = torch.zeros_like(
                    G_i, device=factor_device, dtype=torch.float32
                )
            new_A[name].add_(A_i.float())
            new_G[name].add_(G_i.float())
            new_counts[name] += 1

        for key in set(new_A.keys()) | set(new_G.keys()):
            if use_ema:
                if new_A[key] is not None:
                    if self._param_A[key] is None:
                        self._param_A[key] = new_A[key].clone()
                    else:
                        self._param_A[key].mul_(1 - ema_decay).add_(
                            new_A[key], alpha=ema_decay
                        )
                if new_G[key] is not None:
                    if self._param_G[key] is None:
                        self._param_G[key] = new_G[key].clone()
                    else:
                        self._param_G[key].mul_(1 - ema_decay).add_(
                            new_G[key], alpha=ema_decay
                        )
                self._param_counts[key] = 1
            else:
                if new_A[key] is not None:
                    if self._param_A[key] is None:
                        self._param_A[key] = new_A[key].clone()
                    else:
                        self._param_A[key].add_(new_A[key])
                if new_G[key] is not None:
                    if self._param_G[key] is None:
                        self._param_G[key] = new_G[key].clone()
                    else:
                        self._param_G[key].add_(new_G[key])
                self._param_counts[key] += new_counts[key]

    def _precond_params_generic(self):
        with torch.no_grad():
            for name, p in self._iter_target_params():
                if p.grad is None:
                    continue
                factor_device = self._device_for_key(name, p.device)

                if self.small_matrices:
                    A_avg, G_avg = self._compute_cov_from_raw(name, factor_device)
                    if A_avg is None or G_avg is None:
                        gW = p.grad.to(factor_device)
                        if gW.ndim == 4:
                            gW_mat = gW.flatten(start_dim=1)
                        else:
                            gW_mat = gW
                        A_avg = gW_mat.T @ gW_mat
                        G_avg = gW_mat @ gW_mat.T
                    mu_A, A_avg = _mean_eig_scale(A_avg, 1)
                    mu_G, G_avg = _mean_eig_scale(G_avg, 1)
                else:
                    A_sum = self._param_A.get(name)
                    G_sum = self._param_G.get(name)
                    if A_sum is None or G_sum is None:
                        continue
                    mu_A, A_avg = _mean_eig_scale(A_sum, self._param_counts[name])
                    mu_G, G_avg = _mean_eig_scale(G_sum, self._param_counts[name])
                    A_avg = A_avg.to(factor_device)
                    G_avg = G_avg.to(factor_device)

                lam_A = max(self.rel_eps * mu_A, 1e-5)
                lam_G = max(self.rel_eps * mu_G, 1e-5)

                gW = p.grad.to(factor_device)
                if gW.ndim == 4:
                    gW_mat = gW.flatten(start_dim=1)
                else:
                    gW_mat = gW

                if self.short_side:
                    if gW_mat.shape[1] <= gW_mat.shape[0]:
                        L_A, _ = _chol_with_adaptive_damping(A_avg, lam_A)
                        A_inv = _chol_inv(L_A)
                        gW_new = gW_mat @ A_inv
                    else:
                        L_G, _ = _chol_with_adaptive_damping(G_avg, lam_G)
                        G_inv = _chol_inv(L_G)
                        gW_new = G_inv @ gW_mat
                else:
                    L_A, _ = _chol_with_adaptive_damping(A_avg, lam_A)
                    L_G, _ = _chol_with_adaptive_damping(G_avg, lam_G)
                    A_inv = _chol_inv(L_A)
                    G_inv = _chol_inv(L_G)
                    # assert that all tensors have the same dtype before matmul
                    try:
                        assert A_inv.dtype == gW_mat.dtype == G_inv.dtype
                    except AssertionError:
                        print(
                            f"Warning: dtype mismatch among A_inv ({A_inv.dtype}), gW_mat ({gW_mat.dtype}), and G_inv ({G_inv.dtype}). Ensure that Model (and thus its perametors) are being cast to a consistent dtype."
                        )

                    gW_new = G_inv @ gW_mat @ A_inv

                p.grad.copy_(gW_new.view_as(gW).to(p.grad.device))

    def _init_kfac_syncer(self, kfac_sync_kwargs):
        kwargs = {} if kfac_sync_kwargs is None else dict(kfac_sync_kwargs)
        self._kfac_syncer = KfacGradientMaker(self.model, self.config, **kwargs)

    def _setup_kfac_syncer_calls(self):
        if self._kfac_syncer is None:
            return
        self._kfac_syncer.setup_model_call(
            self._model_fn, *self._model_args, **self._model_kwargs
        )
        if self._loss_fn is None:
            self._kfac_syncer.setup_loss_repr(self._dummy_loss)
        else:
            self._kfac_syncer.setup_loss_call(
                self._loss_fn, *self._loss_fn_args, **self._loss_fn_kwargs
            )
        self._kfac_syncer.setup_logits_repr(self._dummy_logits)

    def update_curvature(self):
        # Generic mode: precondition all 2D/4D weights or regex-matched params.
        if self._use_generic_precond():
            # small_matrices=True: store raw activations/outgrads and build A/G on demand.
            if self.small_matrices:
                """ISSUE: when using small_matrices mode,
                the curvature estimate is effectively a 1-sample approximation regardless of how many training samples you pass through the loop.
                This is because the raw activations and gradients are overwritten in self._install_io_hooks() every pass
                """
                self._install_io_hooks()
                self.forward()
                self.backward()
                self._clear_io_hooks()
            else:
                # Use grad-gram to accumulate A/G directly from parameter gradients.
                self.forward()
                self.backward()
                self._accumulate_param_factors_from_grads()
            return
        # Delegate curvature computation to internal KfacGradientMaker, then extract small factors.
        if self.use_kfac_curvature_update:
            if self._kfac_syncer is None:
                self._init_kfac_syncer(None)
            self._setup_kfac_syncer_calls()
            self._kfac_syncer.update_curvature()
            self._extract_small_factors_from_fisher(self.config.ema_decay)
            return
        # Default small-KFAC path: use gradient Gram for small-side factors.
        if self.use_grad_gram:
            self.forward()
            self.backward()
            self._accumulate_small_factors_adv()
            return

        # Fallback: compute Fisher stats and extract small-side factors.
        config = self.config
        scale = self.scale
        ema_decay = config.ema_decay
        accumulate = ema_decay != 1
        if ema_decay != _INVALID_EMA_DECAY:
            scale *= ema_decay

        self.delegate_forward_and_backward(
            self.fisher_maker,
            data_size=config.data_size,
            scale=scale,
            accumulate=accumulate,
            calc_loss_grad=False,
            calc_inv=False,
            damping=config.damping,
        )
        self._extract_small_factors_from_fisher(ema_decay)

    def update_preconditioner(self):
        if self._use_generic_precond():
            if self.use_kfac_preconditioner_update and self._kfac_syncer is not None:
                self._kfac_syncer.update_preconditioner(
                    **self._kfac_preconditioner_kwargs
                )
            return

        if self.use_kfac_preconditioner_update and self._kfac_syncer is not None:
            self._kfac_syncer.update_preconditioner(**self._kfac_preconditioner_kwargs)

        if not self.use_small_cache:
            return

        self._cache_mlp.clear()
        self._cache_qkv.clear()
        for layer_idx in self._layers_with_factors():
            cache = self._build_cache_for_layer_adv(layer_idx)
            if cache is not None:
                self._cache_mlp[layer_idx] = cache
        for layer_idx, proj in self._qkv_with_factors():
            cache = self._build_cache_for_qkv(layer_idx, proj)
            if cache is not None:
                self._cache_qkv[(layer_idx, proj)] = cache

    def precondition(self):
        if self._use_generic_precond():
            self._precond_params_generic()
            return

        if self._kfac_syncer is not None and dist.is_initialized():
            self._kfac_syncer.sync_grad_pre_precondition(enabled=True, async_op=False)

        with torch.no_grad():
            for layer_idx in self._layers_with_factors():
                if self.use_small_cache and layer_idx in self._cache_mlp:
                    self._precond_with_cache_adv(layer_idx, self._cache_mlp[layer_idx])
                else:
                    self._precond_for_layer(layer_idx)
            for layer_idx, proj in self._qkv_with_factors():
                cache_key = (layer_idx, proj)
                if self.use_small_cache and cache_key in self._cache_qkv:
                    self._precond_qkv_with_cache(
                        layer_idx, proj, self._cache_qkv[cache_key]
                    )
                else:
                    self._precond_for_qkv(layer_idx, proj)

        if self._kfac_syncer is not None and dist.is_initialized():
            self._kfac_syncer.sync_grad_post_precondition(enabled=True, async_op=False)

    def clear_curvature_cache(self, force_gc=True):
        """Clear all accumulated curvature matrices (A_u, G_d, A_qkv, G_qkv) and counts.

        Args:
            force_gc: If True, force garbage collection and empty CUDA cache to release GPU memory.
        """
        # Delete tensors explicitly before clearing dictionaries
        for key in list(self.A_u_dict.keys()):
            if self.A_u_dict[key] is not None:
                del self.A_u_dict[key]
        for key in list(self.G_d_dict.keys()):
            if self.G_d_dict[key] is not None:
                del self.G_d_dict[key]
        for key in list(self.A_qkv_dict.keys()):
            if self.A_qkv_dict[key] is not None:
                del self.A_qkv_dict[key]
        for key in list(self.G_qkv_dict.keys()):
            if self.G_qkv_dict[key] is not None:
                del self.G_qkv_dict[key]

        self.A_u_dict.clear()
        self.G_d_dict.clear()
        self.counts.clear()
        self.A_qkv_dict.clear()
        self.G_qkv_dict.clear()
        self.qkv_counts.clear()

        # Also clear generic parameter factors if they exist
        if hasattr(self, "_param_A"):
            for key in list(self._param_A.keys()):
                if self._param_A[key] is not None:
                    del self._param_A[key]
            self._param_A.clear()
        if hasattr(self, "_param_G"):
            for key in list(self._param_G.keys()):
                if self._param_G[key] is not None:
                    del self._param_G[key]
            self._param_G.clear()
        if hasattr(self, "_param_counts"):
            self._param_counts.clear()

        if force_gc:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        print("Cleared all curvature matrices and counts")

    def clear_preconditioner_cache(self, force_gc=True):
        """Clear all cached preconditioner matrices (Cholesky factors and inverses).

        Args:
            force_gc: If True, force garbage collection and empty CUDA cache to release GPU memory.
        """
        # Delete cached tensors explicitly
        for key in list(self._cache_mlp.keys()):
            if self._cache_mlp[key] is not None:
                for tensor_key in self._cache_mlp[key]:
                    if isinstance(self._cache_mlp[key][tensor_key], torch.Tensor):
                        del self._cache_mlp[key][tensor_key]
                del self._cache_mlp[key]

        for key in list(self._cache_qkv.keys()):
            if self._cache_qkv[key] is not None:
                for tensor_key in self._cache_qkv[key]:
                    if isinstance(self._cache_qkv[key][tensor_key], torch.Tensor):
                        del self._cache_qkv[key][tensor_key]
                del self._cache_qkv[key]

        self._cache_mlp.clear()
        self._cache_qkv.clear()

        if force_gc:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        print("Cleared all preconditioner cache")

    def clear_all_cache(self, force_gc=True):
        """Clear both curvature and preconditioner caches and force GPU memory release.

        Args:
            force_gc: If True, force garbage collection and empty CUDA cache to release GPU memory.
        """
        self.clear_curvature_cache(force_gc=False)
        self.clear_preconditioner_cache(force_gc=False)

        if force_gc:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        print("Cleared all caches (curvature + preconditioner)")
        if force_gc:
            print("Forced garbage collection and emptied CUDA cache")

    def _layers_with_factors(self):
        keys = set(self.A_u_dict.keys()) | set(self.G_d_dict.keys())
        for layer_idx in sorted(keys):
            if self._target_layers is not None and layer_idx not in self._target_layers:
                continue
            if "up_proj" not in self._proj_names or "down_proj" not in self._proj_names:
                continue
            if self.A_u_dict[layer_idx] is None or self.G_d_dict[layer_idx] is None:
                continue
            if self.counts[layer_idx] <= 0:
                continue
            yield layer_idx

    def _qkv_with_factors(self):
        keys = set(self.A_qkv_dict.keys()) | set(self.G_qkv_dict.keys())
        for layer_idx, proj in sorted(keys):
            if self._target_layers is not None and layer_idx not in self._target_layers:
                continue
            if proj not in self._proj_names:
                continue
            if (
                self.A_qkv_dict[(layer_idx, proj)] is None
                or self.G_qkv_dict[(layer_idx, proj)] is None
            ):
                continue
            if self.qkv_counts[(layer_idx, proj)] <= 0:
                continue
            yield layer_idx, proj

    def _extract_small_factors_from_fisher(self, ema_decay):
        if self.fisher_maker is None:
            return
        fisher_attr = self.fisher_maker.config.fisher_attr
        use_ema = ema_decay != _INVALID_EMA_DECAY
        named_modules = dict(self.model.named_modules())

        for name, _p in self.model.named_parameters():
            match = (
                _NAME_PAT_MLP.match(name)
                or _NAME_PAT_Q.match(name)
                or _NAME_PAT_K.match(name)
                or _NAME_PAT_V.match(name)
            )
            if not match:
                continue
            layer = int(match.group(1))
            proj = match.group(2)
            if proj not in self._proj_names:
                continue
            module_name = name.rsplit(".", 1)[0]
            module = named_modules.get(module_name)
            if module is None:
                continue
            matrix = getattr(module, fisher_attr, None)
            if matrix is None or not matrix.has_kron:
                continue

            if proj == "up_proj" and matrix.kron.has_A:
                small = matrix.kron.A.detach().float()
                factor_device = self._device_for_key(f"mlp.{layer}", small.device)
                small = small.to(factor_device)
                self._accumulate_small(self.A_u_dict, layer, small, use_ema, ema_decay)
            elif proj == "down_proj" and matrix.kron.has_B:
                small = matrix.kron.B.detach().float()
                factor_device = self._device_for_key(f"mlp.{layer}", small.device)
                small = small.to(factor_device)
                self._accumulate_small(self.G_d_dict, layer, small, use_ema, ema_decay)
            elif proj in ("q_proj", "k_proj", "v_proj"):
                if matrix.kron.has_A:
                    small = matrix.kron.A.detach().float()
                    factor_device = self._device_for_key(
                        f"qkv.{layer}.{proj}", small.device
                    )
                    small = small.to(factor_device)
                    self._accumulate_small_qkv(
                        self.A_qkv_dict, layer, proj, small, use_ema, ema_decay
                    )
                if matrix.kron.has_B:
                    small = matrix.kron.B.detach().float()
                    factor_device = self._device_for_key(
                        f"qkv.{layer}.{proj}", small.device
                    )
                    small = small.to(factor_device)
                    self._accumulate_small_qkv(
                        self.G_qkv_dict, layer, proj, small, use_ema, ema_decay
                    )

            if hasattr(module, fisher_attr):
                delattr(module, fisher_attr)

    def _accumulate_small(self, storage, layer, value, use_ema, ema_decay):
        if use_ema:
            if storage[layer] is None:
                storage[layer] = value.clone()
            else:
                storage[layer].mul_(1 - ema_decay).add_(value, alpha=ema_decay)
            self.counts[layer] = 1
        else:
            if storage[layer] is None:
                storage[layer] = value.clone()
            else:
                storage[layer].add_(value)
            self.counts[layer] += 1

    def _accumulate_small_qkv(self, storage, layer, proj, value, use_ema, ema_decay):
        key = (layer, proj)
        if use_ema:
            if storage[key] is None:
                storage[key] = value.clone()
            else:
                storage[key].mul_(1 - ema_decay).add_(value, alpha=ema_decay)
            self.qkv_counts[key] = 1
        else:
            if storage[key] is None:
                storage[key] = value.clone()
            else:
                storage[key].add_(value)
            self.qkv_counts[key] += 1

    def _set_precond_grad(self, param, grad_value):
        """Helper method to set preconditioned gradient with proper dtype conversion."""
        if param.grad is None:
            return
        # Convert to the original gradient's dtype to avoid dtype mismatch
        grad_value = grad_value.to(dtype=param.grad.dtype)
        param.grad.copy_(grad_value)

    def _precond_for_layer(self, layer_idx: int):
        if not self.use_woodbury:
            self._precond_for_layer_small_only(layer_idx)
            return
        if self._factor_devices:
            self._precond_for_layer_woodbury_distributed(layer_idx)
        else:
            self._precond_for_layer_woodbury_local(layer_idx)

    def _precond_for_layer_woodbury_local(self, layer_idx: int):
        named = dict(self.model.named_parameters())
        key_u = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        key_d = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if key_u not in named or key_d not in named:
            return

        W_u = named[key_u]
        W_d = named[key_d]
        if W_u.grad is None or W_d.grad is None:
            return

        Au_sum = self.A_u_dict[layer_idx]
        Gd_sum = self.G_d_dict[layer_idx]
        if Au_sum is None or Gd_sum is None:
            return

        factor_device = Au_sum.device
        Au_sum = Au_sum.to(factor_device)
        Gd_sum = Gd_sum.to(factor_device)

        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.counts[layer_idx])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.counts[layer_idx])

        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)

        W_u_local = W_u.detach().to(factor_device, dtype=Au_avg.dtype)
        W_d_local = W_d.detach().to(factor_device, dtype=Gd_avg.dtype)
        WuT_Wu = (W_u_local.T @ W_u_local).to(Au_avg.dtype)
        Wd_WdT = (W_d_local @ W_d_local.T).to(Gd_avg.dtype)

        tr_Ad = torch.einsum("ij,ij->", Au_avg, WuT_Wu).item()
        tr_Gu = torch.einsum("ij,ij->", Gd_avg, Wd_WdT).item()

        h_mlp = W_u.size(0)
        lam_Ad = max(self.rel_eps * (tr_Ad / h_mlp), 1e-5)
        lam_Gu = max(self.rel_eps * (tr_Gu / h_mlp), 1e-5)

        L_Au_raw, _ = _chol_with_adaptive_damping(
            Au_avg, max(1e-5, lam_Au * self.min_damping)
        )
        L_Gd_raw, _ = _chol_with_adaptive_damping(
            Gd_avg, max(1e-5, lam_Gd * self.min_damping)
        )
        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg, lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg, lam_Gd)

        gW_d = W_d.grad.to(factor_device, dtype=Au_avg.dtype)
        gW_u = W_u.grad.to(factor_device, dtype=Au_avg.dtype)

        gW_d_new = _precond_down_full(gW_d, W_u_local, L_Au_raw, L_Gd_damped, lam_Ad)
        gW_u_new = _precond_up_full(gW_u, W_d_local, L_Au_damped, L_Gd_raw, lam_Gu)
        self._set_precond_grad(W_d, gW_d_new.to(W_d.device))
        self._set_precond_grad(W_u, gW_u_new.to(W_u.device))

    def _precond_for_layer_woodbury_distributed(self, layer_idx: int):
        named = dict(self.model.named_parameters())
        key_u = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        key_d = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if key_u not in named or key_d not in named:
            return

        W_u = named[key_u]
        W_d = named[key_d]
        if W_u.grad is None or W_d.grad is None:
            return

        Au_sum = self.A_u_dict[layer_idx]
        Gd_sum = self.G_d_dict[layer_idx]
        if Au_sum is None or Gd_sum is None:
            return

        factor_device = self._device_for_key(f"mlp.{layer_idx}", W_u.device)
        Au_sum = Au_sum.to(factor_device)
        Gd_sum = Gd_sum.to(factor_device)

        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.counts[layer_idx])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.counts[layer_idx])

        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)

        W_u_local = W_u.detach().to(factor_device, dtype=Au_avg.dtype)
        W_d_local = W_d.detach().to(factor_device, dtype=Gd_avg.dtype)
        WuT_Wu = (W_u_local.T @ W_u_local).to(Au_avg.dtype)
        Wd_WdT = (W_d_local @ W_d_local.T).to(Gd_avg.dtype)

        tr_Ad = torch.einsum("ij,ij->", Au_avg, WuT_Wu).item()
        tr_Gu = torch.einsum("ij,ij->", Gd_avg, Wd_WdT).item()

        h_mlp = W_u.size(0)
        lam_Ad = max(self.rel_eps * (tr_Ad / h_mlp), self.min_damping)
        lam_Gu = max(self.rel_eps * (tr_Gu / h_mlp), self.min_damping)

        L_Au_raw, _ = _chol_with_adaptive_damping(
            Au_avg, max(1e-5, lam_Au * self.min_damping)
        )
        L_Gd_raw, _ = _chol_with_adaptive_damping(
            Gd_avg, max(1e-5, lam_Gd * self.min_damping)
        )
        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg, lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg, lam_Gd)

        gW_d = W_d.grad.to(factor_device, dtype=Au_avg.dtype)
        gW_u = W_u.grad.to(factor_device, dtype=Au_avg.dtype)

        gW_d_new = _precond_down_full(gW_d, W_u_local, L_Au_raw, L_Gd_damped, lam_Ad)
        gW_u_new = _precond_up_full(gW_u, W_d_local, L_Au_damped, L_Gd_raw, lam_Gu)
        self._set_precond_grad(W_d, gW_d_new.to(W_d.device))
        self._set_precond_grad(W_u, gW_u_new.to(W_u.device))

    def _precond_with_cache_adv(self, layer_idx: int, cache: Dict[str, torch.Tensor]):
        if cache.get("type") == "small_only":
            self._precond_with_cache_small_only(layer_idx, cache)
            return
        self._precond_with_cache_woodbury(layer_idx, cache)

    def _build_cache_for_layer_adv(self, layer_idx: int):
        if self.use_woodbury:
            return self._build_cache_for_layer_woodbury(layer_idx)
        named = dict(self.model.named_parameters())
        key_u = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        key_d = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if key_u not in named or key_d not in named:
            return None
        Au_sum = self.A_u_dict[layer_idx]
        Gd_sum = self.G_d_dict[layer_idx]
        if Au_sum is None or Gd_sum is None:
            return None
        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.counts[layer_idx])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.counts[layer_idx])
        lam_Au = max(self.rel_eps * mu_Au, 1e-5)
        lam_Gd = max(self.rel_eps * mu_Gd, 1e-5)
        factor_device = self._device_for_key(f"mlp.{layer_idx}", Au_avg.device)
        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg.to(factor_device), lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg.to(factor_device), lam_Gd)
        return {
            "type": "small_only",
            "L_Au_damped": L_Au_damped,
            "L_Gd_damped": L_Gd_damped,
            "key_u": key_u,
            "key_d": key_d,
        }

    def _build_cache_for_layer_woodbury(self, layer_idx: int):
        named = dict(self.model.named_parameters())
        key_u = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        key_d = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if key_u not in named or key_d not in named:
            return None

        W_u = named[key_u]
        W_d = named[key_d]
        Au_sum = self.A_u_dict[layer_idx]
        Gd_sum = self.G_d_dict[layer_idx]
        if Au_sum is None or Gd_sum is None:
            return None

        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.counts[layer_idx])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.counts[layer_idx])

        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)

        factor_device = self._device_for_key(f"mlp.{layer_idx}", W_u.device)
        Au_avg = Au_avg.to(factor_device)
        Gd_avg = Gd_avg.to(factor_device)
        WuT_Wu = (W_u.detach().to(factor_device).T @ W_u.detach().to(factor_device)).to(
            Au_avg.dtype
        )
        Wd_WdT = (W_d.detach().to(factor_device) @ W_d.detach().to(factor_device).T).to(
            Gd_avg.dtype
        )

        tr_Ad = torch.einsum("ij,ij->", Au_avg, WuT_Wu).item()
        tr_Gu = torch.einsum("ij,ij->", Gd_avg, Wd_WdT).item()

        h_mlp = W_u.size(0)
        lam_Ad = max(self.rel_eps * (tr_Ad / h_mlp), 1e-5)
        lam_Gu = max(self.rel_eps * (tr_Gu / h_mlp), 1e-5)

        L_Au_raw, _ = _chol_with_adaptive_damping(
            Au_avg, max(1e-5, lam_Au * self.min_damping)
        )
        L_Gd_raw, _ = _chol_with_adaptive_damping(
            Gd_avg, max(1e-5, lam_Gd * self.min_damping)
        )
        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg, lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg, lam_Gd)

        return {
            "L_Au_raw": L_Au_raw,
            "L_Gd_raw": L_Gd_raw,
            "L_Au_damped": L_Au_damped,
            "L_Gd_damped": L_Gd_damped,
            "lam_Ad": lam_Ad,
            "lam_Gu": lam_Gu,
            "key_u": key_u,
            "key_d": key_d,
        }

    def _precond_with_cache_small_only(
        self, layer_idx: int, cache: Dict[str, torch.Tensor]
    ):
        named = dict(self.model.named_parameters())
        key_u = cache["key_u"]
        key_d = cache["key_d"]
        if key_u not in named or key_d not in named:
            return
        W_u = named[key_u]
        W_d = named[key_d]
        if W_u.grad is None or W_d.grad is None:
            return
        factor_device = cache["L_Au_damped"].device
        gW_d = W_d.grad.to(factor_device)
        gW_u = W_u.grad.to(factor_device)
        Au_inv = _chol_inv(cache["L_Au_damped"])
        Gd_inv = _chol_inv(cache["L_Gd_damped"])
        gW_d_new = Gd_inv @ gW_d
        gW_u_new = gW_u @ Au_inv
        self._set_precond_grad(W_d, gW_d_new.to(W_d.device))
        self._set_precond_grad(W_u, gW_u_new.to(W_u.device))

    def _precond_with_cache_woodbury(
        self, layer_idx: int, cache: Dict[str, torch.Tensor]
    ):
        named = dict(self.model.named_parameters())
        key_u = cache["key_u"]
        key_d = cache["key_d"]
        if key_u not in named or key_d not in named:
            return

        W_u = named[key_u]
        W_d = named[key_d]
        if W_u.grad is None or W_d.grad is None:
            return

        factor_device = cache["L_Au_raw"].device
        gW_d = W_d.grad.to(factor_device, dtype=cache["L_Au_raw"].dtype)
        gW_u = W_u.grad.to(factor_device, dtype=cache["L_Au_raw"].dtype)
        W_u_local = W_u.detach().to(factor_device, dtype=cache["L_Au_raw"].dtype)
        W_d_local = W_d.detach().to(factor_device, dtype=cache["L_Gd_raw"].dtype)

        gW_d_new = _precond_down_full(
            gW_d,
            W_u_local,
            cache["L_Au_raw"],
            cache["L_Gd_damped"],
            cache["lam_Ad"],
        )
        gW_u_new = _precond_up_full(
            gW_u,
            W_d_local,
            cache["L_Au_damped"],
            cache["L_Gd_raw"],
            cache["lam_Gu"],
        )
        self._set_precond_grad(W_d, gW_d_new.to(W_d.device))
        self._set_precond_grad(W_u, gW_u_new.to(W_u.device))

    def _precond_for_qkv(self, layer_idx: int, proj: str):
        named = dict(self.model.named_parameters())
        key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
        if key not in named:
            return

        W = named[key]
        if W.grad is None:
            return

        Au_sum = self.A_qkv_dict[(layer_idx, proj)]
        Gd_sum = self.G_qkv_dict[(layer_idx, proj)]
        if Au_sum is None or Gd_sum is None:
            return
        factor_device = Au_sum.device

        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.qkv_counts[(layer_idx, proj)])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.qkv_counts[(layer_idx, proj)])

        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)

        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg.to(factor_device), lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg.to(factor_device), lam_Gd)

        Au_inv = _chol_inv(L_Au_damped)
        Gd_inv = _chol_inv(L_Gd_damped)
        gW = W.grad.to(factor_device)
        if gW.dtype != Gd_inv.dtype:
            gW = gW.to(dtype=Gd_inv.dtype)
        gW_new = Gd_inv @ gW @ Au_inv
        self._set_precond_grad(W, gW_new.to(W.device))

    def _build_cache_for_qkv(self, layer_idx: int, proj: str):
        named = dict(self.model.named_parameters())
        key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
        if key not in named:
            return None
        Au_sum = self.A_qkv_dict[(layer_idx, proj)]
        Gd_sum = self.G_qkv_dict[(layer_idx, proj)]
        if Au_sum is None or Gd_sum is None:
            return None
        factor_device = self._device_for_key(f"qkv.{layer_idx}.{proj}", Au_sum.device)
        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.qkv_counts[(layer_idx, proj)])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.qkv_counts[(layer_idx, proj)])
        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)
        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg.to(factor_device), lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg.to(factor_device), lam_Gd)
        return {
            "L_Au_damped": L_Au_damped,
            "L_Gd_damped": L_Gd_damped,
            "key": key,
        }

    def _precond_qkv_with_cache(
        self, layer_idx: int, proj: str, cache: Dict[str, torch.Tensor]
    ):
        named = dict(self.model.named_parameters())
        key = cache["key"]
        if key not in named:
            return
        W = named[key]
        if W.grad is None:
            return
        factor_device = cache["L_Au_damped"].device
        gW = W.grad.to(factor_device)
        Au_inv = _chol_inv(cache["L_Au_damped"])
        Gd_inv = _chol_inv(cache["L_Gd_damped"])
        gW_new = Gd_inv @ gW @ Au_inv
        self._set_precond_grad(W, gW_new.to(W.device))

    def _accumulate_small_factors_adv(self):
        ema_decay = self.config.ema_decay
        use_ema = ema_decay != _INVALID_EMA_DECAY

        new_A_u = defaultdict(lambda: None)
        new_G_d = defaultdict(lambda: None)
        new_counts = defaultdict(int)

        new_A_qkv = defaultdict(lambda: None)
        new_G_qkv = defaultdict(lambda: None)
        new_qkv_counts = defaultdict(int)

        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            match = (
                _NAME_PAT_MLP.match(name)
                or _NAME_PAT_Q.match(name)
                or _NAME_PAT_K.match(name)
                or _NAME_PAT_V.match(name)
            )
            if not match:
                continue
            layer = int(match.group(1))
            proj = match.group(2)
            if proj not in self._proj_names:
                continue

            if proj in ("q_proj", "k_proj", "v_proj"):
                key_name = f"qkv.{layer}.{proj}"
            else:
                key_name = f"mlp.{layer}"
            factor_device = self._device_for_key(key_name, p.device)
            autocast_enabled = factor_device.type == "cuda" and self.use_amp

            with torch.amp.autocast(
                device_type=factor_device.type,
                enabled=autocast_enabled,
                dtype=amp_dtype,
            ):
                gW = p.grad.detach().to(factor_device)
                if self.normalize:
                    nd = gW.norm()
                    if nd > 0:
                        gW = gW / nd

                if proj == "down_proj":
                    Gd_i = gW @ gW.T
                    if new_G_d[layer] is None:
                        new_G_d[layer] = torch.zeros_like(
                            Gd_i, device=factor_device, dtype=torch.float32
                        )
                    new_G_d[layer].add_(Gd_i.float())
                    new_counts[layer] += 1

                elif proj == "up_proj":
                    Au_i = gW.T @ gW
                    if new_A_u[layer] is None:
                        new_A_u[layer] = torch.zeros_like(
                            Au_i, device=factor_device, dtype=torch.float32
                        )
                    new_A_u[layer].add_(Au_i.float())
                    new_counts[layer] += 1

                elif proj in ("q_proj", "k_proj", "v_proj"):
                    Au_i = gW.T @ gW
                    Gd_i = gW @ gW.T
                    key = (layer, proj)
                    if new_A_qkv[key] is None:
                        new_A_qkv[key] = torch.zeros_like(
                            Au_i, device=factor_device, dtype=torch.float32
                        )
                    if new_G_qkv[key] is None:
                        new_G_qkv[key] = torch.zeros_like(
                            Gd_i, device=factor_device, dtype=torch.float32
                        )
                    new_A_qkv[key].add_(Au_i.float())
                    new_G_qkv[key].add_(Gd_i.float())
                    new_qkv_counts[key] += 1

        for layer in set(new_A_u.keys()) | set(new_G_d.keys()):
            if use_ema:
                if new_A_u[layer] is not None:
                    if self.A_u_dict[layer] is None:
                        self.A_u_dict[layer] = new_A_u[layer].clone()
                    else:
                        self.A_u_dict[layer].mul_(1 - ema_decay).add_(
                            new_A_u[layer], alpha=ema_decay
                        )
                if new_G_d[layer] is not None:
                    if self.G_d_dict[layer] is None:
                        self.G_d_dict[layer] = new_G_d[layer].clone()
                    else:
                        self.G_d_dict[layer].mul_(1 - ema_decay).add_(
                            new_G_d[layer], alpha=ema_decay
                        )
                self.counts[layer] = 1
            else:
                if new_A_u[layer] is not None:
                    if self.A_u_dict[layer] is None:
                        self.A_u_dict[layer] = new_A_u[layer].clone()
                    else:
                        self.A_u_dict[layer].add_(new_A_u[layer])
                if new_G_d[layer] is not None:
                    if self.G_d_dict[layer] is None:
                        self.G_d_dict[layer] = new_G_d[layer].clone()
                    else:
                        self.G_d_dict[layer].add_(new_G_d[layer])
                self.counts[layer] += new_counts[layer]

        for key in set(new_A_qkv.keys()) | set(new_G_qkv.keys()):
            if use_ema:
                if new_A_qkv[key] is not None:
                    if self.A_qkv_dict[key] is None:
                        self.A_qkv_dict[key] = new_A_qkv[key].clone()
                    else:
                        self.A_qkv_dict[key].mul_(1 - ema_decay).add_(
                            new_A_qkv[key], alpha=ema_decay
                        )
                if new_G_qkv[key] is not None:
                    if self.G_qkv_dict[key] is None:
                        self.G_qkv_dict[key] = new_G_qkv[key].clone()
                    else:
                        self.G_qkv_dict[key].mul_(1 - ema_decay).add_(
                            new_G_qkv[key], alpha=ema_decay
                        )
                self.qkv_counts[key] = 1
            else:
                if new_A_qkv[key] is not None:
                    if self.A_qkv_dict[key] is None:
                        self.A_qkv_dict[key] = new_A_qkv[key].clone()
                    else:
                        self.A_qkv_dict[key].add_(new_A_qkv[key])
                if new_G_qkv[key] is not None:
                    if self.G_qkv_dict[key] is None:
                        self.G_qkv_dict[key] = new_G_qkv[key].clone()
                    else:
                        self.G_qkv_dict[key].add_(new_G_qkv[key])
                self.qkv_counts[key] += new_qkv_counts[key]

    def _precond_for_layer_small_only(self, layer_idx: int):
        named = dict(self.model.named_parameters())
        key_u = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        key_d = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if key_u not in named or key_d not in named:
            return

        W_u = named[key_u]
        W_d = named[key_d]
        if W_u.grad is None or W_d.grad is None:
            return

        Au_sum = self.A_u_dict[layer_idx]
        Gd_sum = self.G_d_dict[layer_idx]
        if Au_sum is None or Gd_sum is None:
            return
        factor_device = Au_sum.device

        mu_Au, Au_avg = _mean_eig_scale(Au_sum, self.counts[layer_idx])
        mu_Gd, Gd_avg = _mean_eig_scale(Gd_sum, self.counts[layer_idx])

        lam_Au = max(self.rel_eps * mu_Au, self.min_damping)
        lam_Gd = max(self.rel_eps * mu_Gd, self.min_damping)

        L_Au_damped, _ = _chol_with_adaptive_damping(Au_avg.to(factor_device), lam_Au)
        L_Gd_damped, _ = _chol_with_adaptive_damping(Gd_avg.to(factor_device), lam_Gd)

        Au_inv = _chol_inv(L_Au_damped)
        Gd_inv = _chol_inv(L_Gd_damped)

        gW_d = W_d.grad.to(factor_device)
        gW_u = W_u.grad.to(factor_device)
        if gW_d.dtype != Gd_inv.dtype:
            gW_d = gW_d.to(dtype=Gd_inv.dtype)
        if gW_u.dtype != Au_inv.dtype:
            gW_u = gW_u.to(dtype=Au_inv.dtype)
        gW_d_new = Gd_inv @ gW_d
        gW_u_new = gW_u @ Au_inv

        # Check if any grad is too big, trigger debugger
        # gW_d_norm = gW_d_new.norm().item()
        # gW_u_norm = gW_u_new.norm().item()
        # # if gW_d_norm > 1e3 or gW_u_norm > 1e3:
        # import ipdb; ipdb.set_trace()

        self._set_precond_grad(W_d, gW_d_new.to(W_d.device))
        self._set_precond_grad(W_u, gW_u_new.to(W_u.device))

    def save_state(self, path: str):
        state = {
            "A_u_dict": {
                k: v.cpu() if v is not None else None for k, v in self.A_u_dict.items()
            },
            "G_d_dict": {
                k: v.cpu() if v is not None else None for k, v in self.G_d_dict.items()
            },
            "counts": dict(self.counts),
            "A_qkv_dict": {
                k: v.cpu() if v is not None else None
                for k, v in self.A_qkv_dict.items()
            },
            "G_qkv_dict": {
                k: v.cpu() if v is not None else None
                for k, v in self.G_qkv_dict.items()
            },
            "qkv_counts": dict(self.qkv_counts),
        }
        torch.save(state, path)
        print(f"Saved K-FAC state to {path}")

    def load_state(self, path: str):
        state = torch.load(path, map_location="cpu")
        for k, v in state["A_u_dict"].items():
            self.A_u_dict[k] = (
                v.to(self._device_for_key(f"mlp.{k}", v.device))
                if v is not None
                else None
            )
        for k, v in state["G_d_dict"].items():
            self.G_d_dict[k] = (
                v.to(self._device_for_key(f"mlp.{k}", v.device))
                if v is not None
                else None
            )
        self.counts.update(state["counts"])
        for k, v in state["A_qkv_dict"].items():
            self.A_qkv_dict[k] = (
                v.to(self._device_for_key(f"qkv.{k[0]}.{k[1]}", v.device))
                if v is not None
                else None
            )
        for k, v in state["G_qkv_dict"].items():
            self.G_qkv_dict[k] = (
                v.to(self._device_for_key(f"qkv.{k[0]}.{k[1]}", v.device))
                if v is not None
                else None
            )
        self.qkv_counts.update(state["qkv_counts"])
        print(f"Loaded K-FAC state from {path}")
