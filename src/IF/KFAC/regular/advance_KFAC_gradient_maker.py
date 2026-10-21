import torch
from torch import nn
import torch.distributed as dist

from asdl.precondition  import PreconditioningConfig,NaturalGradientMaker
from asdl.fisher import  LOSS_CROSS_ENTROPY
from asdl.matrices import SHAPE_KRON

from asdl import NaturalGradientMaker
from asdl import PreconditioningConfig
from asdl.vector import ParamVector
from asdl.matrices import *
from asdl.symmatrix import *
from asdl.fisher import FISHER_MC, LOSS_CROSS_ENTROPY

from IF.KFAC.matrix_operations import (
    _bias_requires_grad,
    __all__,
)


class AdvanceKFACGradientMaker(NaturalGradientMaker):
    """GradientMaker for calculating the preconditioned gradient by `K-FAC <https://arxiv.org/abs/1503.05671>`_.
    A wrapper class of :ref:`NaturalGradientMaker <natural_gradient_maker>`.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        
        fisher_type (str, optional): The type of Fisher information matrix estimation:
            *"fisher_exact"* (an exact calculation), *"fisher_mc"* (an estimate by MC samples),
            or *"fisher_emp"* (empirical Fisher). (default: *"fisher_mc"*)
            
        loss_type (str, optional): The type of loss on which the Fisher calculation is based:
            *"cross_entropy"* or *"mse"*. This does not have any effect when :obj:`fisher_type = "fisher_emp"`.
            (default: *"cross_entropy"*)
            
        scale (float, optional): The scale of Fisher information matrix. (default: 1.)
        n_mc_samples (int, optional): The number of MC samples used when :obj:`fisher_type = "fisher_mc"`.
            (default: 1)
            
        var (float, optional): The variance of the Gaussian distribution used when
            :obj:`fisher_type = "fisher_mc"` and :obj:`loss_type = "mse"`. (default: 1.)
            
        seed (int, optional): The random seed set every time Fisher matrix is calculated.
            This is useful to make the sampling deterministic when :obj:`fisher_type = "fisher_mc"`
            (default: None)
            
        swift (bool, optional): If True, KfacGradientMaker calculates preconditioned gradients by
            `SKFAC <https://ieeexplore.ieee.org/document/9578481>`_ algorithm. (default: False)
    """
    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, loss_type: str = LOSS_CROSS_ENTROPY,
                 scale: float = 1, grad_scale: float = 1,
                 n_mc_samples: int = 1, var: float = 1., seed: int = None, swift=False):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config,
                         fisher_type=fisher_type,
                         fisher_shape=fisher_shape,
                         loss_type=loss_type,
                         scale=scale,
                         grad_scale=grad_scale,
                         n_mc_samples=n_mc_samples,
                         var=var,
                         seed=seed
                         )
    
    def run(self):
        step = self.state["step"]
        if step==0:
            print("This KFAC precondition with data_parallel instead of the original asdl implementation")
        self._startup()
        # Debugging line, remove if not needed
        if self.do_forward_and_backward(step):
            self.forward()
            self.backward()
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()

        self.precondition_data_parallel()

        self._teardown()

        self.state["step"] += 1

        return self._model_output, self._loss.mean()  # Return model output and mean loss
    
    def precondition_data_parallel(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        """Precondition the gradients for the model parameters.

        This method performs a three-step process to precondition the gradients:
        1. Reduce-Scatter: Averages gradients only for modules each rank owns.
        2. Owner Precondition: Each rank preconditions its assigned modules.
        3. All-Gather: Broadcasts preconditioned gradients from owners to all ranks.
        """
        if grad_scale is None:
            grad_scale = self.grad_scale
        
        # Step 1: Reduce-Scatter - Average gradients only for modules each rank owns
        if self.world_size > 1:
            self._reduce_scatter_gradients_by_partition()
        
        # Step 2: Owner Precondition - Each rank preconditions its assigned modules
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    if not self.is_module_for_inv_and_precondition(module):
                        continue
                    self._precondition_module(module, shape, vectors, grad_scale=grad_scale, use_inv=use_inv)

        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            if fisher is None:
                raise ValueError(f'Fisher of shape {SHAPE_FULL} has not been calculated.')
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            if vectors is None:
                raise ValueError('gradient has not been calculated.')
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)

        # Step 3: All-Gather - Broadcast preconditioned gradients from owners to all ranks
        if self.world_size > 1:
            self._all_gather_preconditioned_gradients_by_partition()
    
    def _reduce_scatter_gradients_by_partition(self):
        """Step 1: Reduce-scatter gradients so each rank gets averaged gradients for its owned modules."""
        if not dist.is_initialized():
            return
            
        # Group modules by owning rank
        rank_modules = {rank: [] for rank in range(self.world_size)}
        
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                owning_rank = self.partitions[enum_shape][enum_module]
                if self.is_module_for_inv_and_precondition(module):
                    rank_modules[owning_rank].append(module)
        
        # For each rank's modules, perform reduce operation to average gradients
        for rank, modules in rank_modules.items():
            if len(modules) == 0:
                continue
                
            # Collect gradients for this rank's modules from all ranks
            grads_for_rank = []
            for module in modules:
                if hasattr(module, 'weight') and module.weight.requires_grad and module.weight.grad is not None:
                    grads_for_rank.append(module.weight.grad)
                if _bias_requires_grad(module) and module.bias.grad is not None:
                    grads_for_rank.append(module.bias.grad)
            
            if len(grads_for_rank) == 0:
                continue
                
            # Pack and reduce to the owning rank (averages gradients across all ranks)
            packed_grad = parameters_to_vector(grads_for_rank)
            dist.reduce(packed_grad, dst=rank, op=dist.ReduceOp.AVG)
            
            # Unpack averaged gradients on the owning rank
            if self.world_rank == rank:
                vector_to_parameters(packed_grad, grads_for_rank)
    
    def _all_gather_preconditioned_gradients_by_partition(self):
        """Step 3: Broadcast preconditioned gradients from owning ranks to all ranks."""
        if not dist.is_initialized():
            return
            
        # Broadcast preconditioned gradients from each owning rank to all other ranks
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                owning_rank = self.partitions[enum_shape][enum_module]
                
                if not self.is_module_for_inv_and_precondition(module):
                    continue
                    
                # Collect gradients to broadcast (these are now preconditioned on the owning rank)
                grads_to_broadcast = []
                if hasattr(module, 'weight') and module.weight.requires_grad and module.weight.grad is not None:
                    grads_to_broadcast.append(module.weight.grad)
                if _bias_requires_grad(module) and module.bias.grad is not None:
                    grads_to_broadcast.append(module.bias.grad)
                
                if len(grads_to_broadcast) == 0:
                    continue
                    
                # Pack and broadcast preconditioned gradients from owning rank
                packed_grad = parameters_to_vector(grads_to_broadcast)
                dist.broadcast(packed_grad, src=owning_rank)
                
                # Unpack preconditioned gradients on all ranks
                vector_to_parameters(packed_grad, grads_to_broadcast)

    def _precondition_module(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
                            grad_scale=None, use_inv=True):
        
        if grad_scale is None:
            grad_scale = self.grad_scale
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        if shape is None:
            raise ValueError(f'No shape is assigned to module: {module}.')
        matrix = self._get_module_symmatrix(module, shape)
        if matrix is None:
            raise ValueError(f'Matrix of shape {shape} for module {module} has not been calculated.')
        if vec_weight is None and module.weight.requires_grad:
            vec_weight = module.weight.grad
        if vec_weight is None:
            raise ValueError(f'weight gradient for module {module} has not been calculated.')
        if _bias_requires_grad(module):
            if vec_bias is None:
                vec_bias = module.bias.grad
            if vec_bias is None:
                raise ValueError(f'bias gradient for module {module} has not been calculated.')
            
        if grad_scale != 1:
            vec_weight.data.mul_(grad_scale)
            if vec_bias is not None:
                vec_bias.data.mul_(grad_scale)
        if not use_inv or matrix.has_inv:
            kwargs = dict(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=use_inv, inplace=True)
            if shape == SHAPE_KFE:
                kwargs['eps'] = self.config.damping
            matrix.mvp(**kwargs)

