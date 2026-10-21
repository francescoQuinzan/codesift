from IF.KFAC.small.advance_small_KFAC_gradient_maker import (
    AdvancedSmallKfacGradientMaker,
)
from typing import Callable, List, Optional
import torch
from transformers import AutoTokenizer
from IF.Utils.test_utils import _prepare_inputs, cross_entropy_for_lm

from uuid import uuid4


class Influences:
    def __init__(
        self,
        gradient_maker: AdvancedSmallKfacGradientMaker,
        tokenizer: AutoTokenizer,
        train_ds: list,
        test_ds: List | str,
        device: torch.device,
        dtype: torch.dtype,
        epsilon=1e-3,
        curvature_loss_fn: Optional[Callable] = None,
        eval_loss_fn: Optional[Callable] = None,
        curvature_loss_context_config: Optional[dict] = None,
        eval_loss_context_config: Optional[dict] = None,
    ):
        self.gm = gradient_maker
        self.epsilon = epsilon

        gm_device = next(self.gm.model.parameters()).device
        self.device = gm_device
        self.dtype = dtype

        self.model = self.gm.model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_ds = self._formatDataset(train_ds)
        self.test_ds = self._formatDataset(test_ds)

        self.curvature_loss_fn = curvature_loss_fn
        self.eval_loss_fn = eval_loss_fn
        self.curvature_loss_context_config = curvature_loss_context_config or None
        self.eval_loss_context_config = eval_loss_context_config or None

        self.conditioned_grads = None

    def _formatDataset(self, dataset):
        """
        ensure that each string in dataset is the length of the max window size of the model
        """
        max_length = self.model.config.max_position_embeddings
        formatted = []
        for text in dataset:
            if len(text) > max_length:
                for i in range(0, len(text), max_length):
                    s = text[i : i + max_length]
                    formatted.append(s.replace("\n", ""))
            else:
                formatted.append(text.replace("\n", ""))

        return formatted

    def _flattenGrads(self, grads):
        return torch.cat([g.view(-1) for g in grads])

    def setupGM(self, enc, labels, loss_fn=None, loss_fn_context=None, text=None):
        self.gm.model.zero_grad(set_to_none=True)

        if loss_fn is not None:
            if not hasattr(loss_fn, "set_context"):
                raise ValueError(
                    "Loss function must have a set_context method to be used as a custom loss function in Influences."
                )
            loss_fn.set_context(
                text,
                enc,
                model=self.model,
                tokenizer=self.tokenizer,
                gen_cfg=loss_fn_context,
            )

        import time
        start = time.perf_counter()
        if loss_fn is None:
            dummy = self.gm.setup_model_call(self.model, **enc, labels=labels)
            self.gm.setup_logits_repr(dummy.logits)
            self.gm.setup_loss_call(cross_entropy_for_lm, dummy.logits, labels)
        else:
            dummy = self.gm.setup_model_call(self.model, **enc)
            self.gm.setup_logits_repr(dummy.logits)
            self.gm.setup_loss_call(loss_fn)
        elapsed = time.perf_counter() - start
        print(f"Routine took {elapsed:.6f} seconds")

    def computeCurvatureAndPrecondition(self):
        for text in self.train_ds:
            self.gm.data_size = len(self.train_ds)

            enc, labels = _prepare_inputs(self.tokenizer, [text], self.device)
            self.setupGM(
                enc,
                labels,
                loss_fn=self.curvature_loss_fn,
                loss_fn_context=self.curvature_loss_context_config,
                text=text,
            )
            self.gm.update_curvature()

        self.gm.update_preconditioner()

    def calculateCandidateLossGradient(self):
        eval_text = self.test_ds[0]
        enc_eval, labels_eval = _prepare_inputs(
            self.tokenizer, [eval_text], self.device
        )

        self.setupGM(
            enc_eval,
            labels_eval,
            loss_fn=self.eval_loss_fn,
            loss_fn_context=self.eval_loss_context_config,
            text=eval_text,
        )
        self.gm.forward()
        self.gm.backward()

    def applyPreconditionToEval(self):
        self.gm.precondition()

    def layer_influence_score(self, prompt_text, loss_fn=None):
        self.influence_scores = {}
        for (pk, pv), (tk, tv) in zip(
            self.conditioned_grads.items(),
            self.functionGradWRTParams(prompt_text, func=loss_fn).items(),
        ):
            assert pk == tk
            pv, tv = self._flattenGrads(pv), self._flattenGrads(tv)

            score = torch.dot(pv, tv).item()
            self.influence_scores[pk] = score
        return self.influence_scores

    def candidateInfluence(self):
        self.computeCurvatureAndPrecondition()
        self.calculateCandidateLossGradient()
        self.applyPreconditionToEval()

        self.conditioned_grads = {
            name: -1 * self.epsilon * p.grad.detach().clone()
            for name, p in self.gm._iter_target_params()
            if p.grad is not None
        }
        return self.conditioned_grads

    def functionGradWRTParams(self, prompt_text, func=None):
        enc, labels = _prepare_inputs(self.tokenizer, [prompt_text], self.device)
        self.setupGM(
            enc,
            labels,
            loss_fn=func,
            loss_fn_context=self.eval_loss_context_config,
            text=prompt_text,
        )
        self.gm.forward()
        self.gm.backward()
        return {
            name: p.grad.detach().clone()
            for name, p in self.gm._iter_target_params()
            if p.grad is not None
        }

    def get_state(self):
        return {
            "conditioned_grads": self.conditioned_grads,
            "epsilon": self.epsilon,
            "model_name": self.model.config._name_or_path,
        }

    def __hash__(self):
        return hash(
            hash(self.gm),
            hash(self.get_state()),
        )

    def save(self, path=None):
        self_hash = self.__hash__()
        gm_path = (
            f"{path}_gm.pt"
            if path
            else f"influence_gm_{uuid4()}{self_hash}.pt"
        )
        self.gm.save_state(gm_path)

        self_save_path = (
            f"{path}_influence.pt"
            if path
            else f"influence_{uuid4()}{self_hash}.pt"
        )
        torch.save(self.get_state(), self_save_path)
        print(
            f"Saved Influences state to {self_save_path} and gradient maker state to {gm_path}"
        )

    def load_state(self, path):
        state = torch.load(path)
        self.conditioned_grads = state["conditioned_grads"]
        self.epsilon = state["epsilon"]