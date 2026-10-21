from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from collections.abc import Sequence, Callable
from tqdm.auto import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataclasses import dataclass
from typing import Dict, Tuple
from transformers import BatchEncoding

import builtins

from detoxify import Detoxify

import subprocess
import json
import tempfile

import os
import shutil

from typing import Iterable, List, Optional
import json
import os
import subprocess
import tempfile

import torch
import torch.nn as nn

from IF.Utils import semgrep_llm_cleaner
from IF.Utils.semgrep_llm_cleaner import clean_and_save
from IF import SEMGREP_INSTRUCTIONS_PATH


# ----------------------------------------------------------------------------- #
# A dict[str, torch.Tensor] with tensor-wide device/dtype utilities             #
# ----------------------------------------------------------------------------- #
class TensorDict(dict[str, torch.Tensor]):
    """
    A dict[str, torch.Tensor] with tensor-wide device/dtype utilities.
    Non-tensor values (if any) are passed through unchanged.
    """

    def to(self, *args, **kwargs) -> "TensorDict":
        """Return a new TensorDict with all tensor values v.to(*args, **kwargs)."""
        out = TensorDict()
        for k, v in self.items():
            out[k] = v.to(*args, **kwargs) if torch.is_tensor(v) else v
        return out

    def to_(self, *args, **kwargs) -> "TensorDict":
        """In-place version of .to(...) (mutates)."""
        for k, v in list(self.items()):
            if torch.is_tensor(v):
                self[k] = v.to(*args, **kwargs)
        return self

    def cpu(self) -> "TensorDict":
        """In-place move to CPU (like nn.Module.cpu())."""
        return self.to_("cpu")

    def cuda(self, device: int | str | None = None, **to_kwargs) -> "TensorDict":
        """
        In-place move to CUDA. device can be int (index) or str like 'cuda:1'.
        Extra kwargs are passed to Tensor.to (e.g., non_blocking=True).
        """
        dev = f"cuda:{device}" if isinstance(device, int) else (device or "cuda")
        return self.to_(dev, **to_kwargs)

    def __iadd__(self, other):
        if isinstance(other, TensorDict):
            # Ensure matching keys
            if self.keys() != other.keys():
                missing_self = other.keys() - self.keys()
                missing_other = self.keys() - other.keys()
                raise KeyError(
                    f"TensorDict key mismatch. "
                    f"Missing in self: {missing_self or 'none'}, "
                    f"missing in other: {missing_other or 'none'}."
                )
            # Add in-place elementwise
            for k in self:
                if torch.is_tensor(self[k]) and torch.is_tensor(other[k]):
                    self[k] = self[k] + other[k]
        else:
            # Scalar or tensor broadcast
            for k in self:
                if torch.is_tensor(self[k]):
                    self[k] = self[k] + other
        return self

    def __add__(self, other):
        out = TensorDict()
        if isinstance(other, TensorDict):
            if self.keys() != other.keys():
                missing_self = other.keys() - self.keys()
                missing_other = self.keys() - other.keys()
                raise KeyError(
                    f"TensorDict key mismatch. "
                    f"Missing in self: {missing_self or 'none'}, "
                    f"missing in other: {missing_other or 'none'}."
                )
            for k in self:
                if torch.is_tensor(self[k]) and torch.is_tensor(other[k]):
                    out[k] = self[k] + other[k]
        else:
            for k in self:
                v = self[k]
                out[k] = v + other if torch.is_tensor(v) else v
        return out

    def __mul__(self, other):
        out = TensorDict()
        if isinstance(other, TensorDict):
            if self.keys() != other.keys():
                raise KeyError("TensorDict key mismatch")
            for k in self:
                out[k] = self[k] * other[k]
        else:
            for k, v in self.items():
                out[k] = v * other if torch.is_tensor(v) else v
        return out

    def __imul__(self, other):
        if isinstance(other, TensorDict):
            if self.keys() != other.keys():
                raise KeyError("TensorDict key mismatch")
            for k in self:
                self[k] = self[k] * other[k]
        else:
            for k in self:
                if torch.is_tensor(self[k]):
                    self[k] = self[k] * other
        return self

    def __truediv__(self, other):
        """Return a new TensorDict with all tensors divided by `other`."""
        out = TensorDict()
        if isinstance(other, TensorDict):
            if self.keys() != other.keys():
                raise KeyError("TensorDict key mismatch")
            for k in self:
                if torch.is_tensor(self[k]) and torch.is_tensor(other[k]):
                    out[k] = self[k] / other[k]
        else:
            for k, v in self.items():
                out[k] = v / other if torch.is_tensor(v) else v
        return out

    def __itruediv__(self, other):
        """In-place division of all tensors by `other`."""
        if isinstance(other, TensorDict):
            if self.keys() != other.keys():
                raise KeyError("TensorDict key mismatch")
            for k in self:
                if torch.is_tensor(self[k]) and torch.is_tensor(other[k]):
                    self[k] = self[k] / other[k]
        else:
            for k in self:
                if torch.is_tensor(self[k]):
                    self[k] = self[k] / other
        return self


# ----------------------------------------------------------------------------- #
# Global cache for reward-model assets                                          #
# ----------------------------------------------------------------------------- #
@dataclass
class RewardAssets:
    """Lightweight container for reward-model resources.

    Attributes
    ----------
    model : object
        A *frozen* Hugging Face sequence-classification model used as a
        reward function (e.g., GRM). Keep this on the chosen device.
    tokenizer : object
        Matching tokenizer for the reward model.
    dtype : torch.dtype
        Floating-point dtype in which the reward model was loaded.
    """

    model: object
    tokenizer: object
    dtype: torch.dtype


# Cache keyed by (model_id, dtype) so we only download/load once per precision.
_REWARD_CACHE: Dict[Tuple[str, torch.dtype], RewardAssets] = {}


def _set_reward_assets(
    model_id: str,
    *,
    dtype: torch.dtype,
    model: object,
    tokenizer: object,
) -> None:
    """Insert (or overwrite) a cache entry for a reward model.

    Notes
    -----
    The dtype is part of the cache key so you can maintain separate copies of
    the same model in different precisions (e.g., float16 and bfloat16).
    """
    key = (model_id, dtype)
    assets = RewardAssets(model=model, tokenizer=tokenizer, dtype=dtype)
    _REWARD_CACHE[key] = assets
    return


def _get_reward_assets(
    model_id: str,
    *,
    dtype: torch.dtype,
) -> RewardAssets | None:
    """Look up a reward-model entry in the global cache.

    Returns
    -------
    RewardAssets | None
        The cached assets if present, otherwise ``None``.
    """
    return _REWARD_CACHE.get((model_id, dtype))


# ----------------------------------------------------------------------------- #
# Loss wrapper base class
# ----------------------------------------------------------------------------- #
class LossWrapper(ABC):
    """Abstract base class for policy-loss computations.

    Responsibilities
    ----------------
    - Holds a *frozen* snapshot of the policy's initial parameters.
    - Prepares common inputs/devices/dtypes.
    - Provides helpers to compute gradients and group them by parameter name.

    Subclasses must implement :meth:`loss`, which returns a scalar loss with a
    live autograd graph (do not call ``.backward()`` inside ``loss``).
    """

    def __init__(
        self,
        prompt: str,
        enc: BatchEncoding,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # ----------------- model / tokenizer / generation config -------------- #
        # Detect if the model was loaded with bitsandbytes quantization (8-bit or 4-bit).
        # Quantized models are already dispatched to devices during `from_pretrained`,
        # and cannot be moved or cast again with `.to()`.
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            self.model = model
        # For standard (non-quantized) models: move to the requested device.
        else:
            self.model = model.to(device, dtype=dtype)
        self.tokenizer = tokenizer
        self.gen_cfg = gen_cfg
        # Fallback to EOS as pad if the tokenizer has no explicit pad token.
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # --------------------------- device / dtype --------------------------- #
        # Default to CUDA if available, otherwise CPU. Allow explicit override.
        # --- Pick the right target device for inputs ---
        if getattr(model, "is_loaded_in_8bit", True) or getattr(
            model, "is_loaded_in_4bit", True
        ):
            # Quantized models: use the device of the input embeddings.
            # This is robust even when the model is sharded across devices.
            self.device = model.get_input_embeddings().weight.device
        elif device is not None:
            # Non-quantized models: use the model's (target) device you already chose.
            self.device = device
        else:
            (
                device
                if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.dtype = dtype

        # ---------------------- parameter snapshots --------------------------- #
        # Keep a detached snapshot of the *initial* parameters for reference.
        self.base_param_dict = {n: p.detach() for n, p in self.model.named_parameters()}
        self.base_params = tuple(self.base_param_dict.values())

        # --------------------------- module index ----------------------------- #
        # Flat dict of all named submodules so we can quickly retrieve layers
        # by prefix (e.g., "model.layers.3.mlp").
        self.named_modules_dict = dict(self.model.named_modules())

        # --------------------------- inputs ---------------------------------- #
        self.prompt = prompt
        # Ensure all encoded inputs are on the model's device.
        self.enc = {k: v.to(self.device) for k, v in enc.items()}
        self.prompt_len = self.enc["input_ids"].shape[1]

    # --------------------------------------------------------------------- #
    # Gradient utilities
    # --------------------------------------------------------------------- #

    def get_layer(self, layer_name: str) -> torch.nn.modules:
        """Return the owning module of a parameter given its *full* name.

        Example
        -------
        ``layer_name = 'model.layers.3.mlp.gate_proj.weight'`` → returns the
        ``'model.layers.3.mlp'`` module.
        """
        prefix = layer_name.rsplit(".", 1)[0]
        return self.named_modules_dict[prefix]

    def grads_grouped(self, grads_flat, params, name_filter: str = "mlp") -> TensorDict:
        """Map a flat gradient vector back to parameter-shaped tensors.

        Parameters
        ----------
        grads_flat : torch.Tensor
            Concatenation of per-parameter gradients (1-D).
        id : str, default "mlp"
            Only include parameters whose *name* contains this substring.

        Returns
        -------
        TensorDict
            ``{full_param_name: grad_tensor}`` filtered by ``id``.
        """
        param_id_set = {builtins.id(q) for q in params}
        param_slices, param_shapes, offset = {}, {}, 0
        for full_name, p in self.model.named_parameters():
            if builtins.id(p) in param_id_set:
                n = p.numel()
                param_slices[full_name] = slice(offset, offset + n)
                param_shapes[full_name] = p.shape
                offset += n

        return TensorDict(
            {
                name: grads_flat[sl].view(param_shapes[name])
                for name, sl in param_slices.items()
                if name_filter in name
            }
        )

    # --------------------------------------------------------------------- #
    # Sampling & REINFORCE core
    # --------------------------------------------------------------------- #

    def reinforce_loss(self) -> tuple[torch.Tensor, list[str]]:
        """Compute a single-sample REINFORCE objective and decoded replies.

        Steps
        -----
        1. Sample from the policy **without** building an autograd graph.
        2. Build labels to compute a causal LM NLL over the generated tokens
           (ignoring pad tokens and, by convention, the prompt tokens).
        3. Forward once more **with** labels to obtain a loss whose graph can be
           used for influence/HvP.

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            ``(-nll_loss, decoded_replies)`` where the negative sign converts NLL
            into a *reward-like* quantity for REINFORCE.
        """
        # 1) Sample (no graph, minimal returns)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **self.enc,
                generation_config=self.gen_cfg,
                return_dict_in_generate=False,
                output_scores=False,
                pad_token_id=self.tokenizer.pad_token_id,  # new add
            )

        # 2) Build labels that align with generated tokens; mask out padding.
        labels = torch.full_like(generated_ids, -100)
        if self.prompt_len > 0:
            labels[:, self.prompt_len :] = generated_ids[:, self.prompt_len :]
        else:
            labels.copy_(generated_ids)
        labels.masked_fill_(generated_ids.eq(self.pad_id), -100)
        attn = generated_ids.ne(self.pad_id).to(dtype=torch.long)

        # 3) Compute NLL with a full autograd graph.
        fw = self.model(
            input_ids=generated_ids,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        # Decode only the assistant portion (post-prompt).
        reply_ids = generated_ids[:, self.prompt_len :].detach()
        replies = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
        # replies = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Proactive cleanup to keep memory in check on CUDA.
        del reply_ids, attn, labels
        torch.cuda.empty_cache()

        return -fw.loss, replies

    # Subclasses *must* supply the scalar loss
    @abstractmethod
    def loss(self, *param_tuple) -> torch.Tensor: ...

    def _grad(self, name_filter: str = "mlp") -> TensorDict:
        """Compute the gradient of the scalar loss wrt policy parameters.

        Returns
        -------
        layer_grads : dict[str, torch.Tensor]
            Gradients reshaped back to their parameter tensors, filtered by
            parameter-name substring ``id``.
        """
        loss, reward = self.loss()
        params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad
            if p.dtype.is_floating_point
            if name_filter in name
        ]
        grads = torch.autograd.grad(
            loss * reward,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grads_flat = torch.cat([g.contiguous().view(-1) for g in grads])
        del grads
        layer_grads = self.grads_grouped(grads_flat, params, name_filter)
        del grads_flat, loss

        return layer_grads, reward

    def grad(
        self, name_filter: str = "mlp", num_samples: int = 1
    ) -> tuple[TensorDict, torch.tensor]:
        """Compute the gradient mean of the scalar loss wrt policy parameters.

        Returns
        -------
        layer_grads : dict[str, torch.Tensor]
            Gradients reshaped back to their parameter tensors, filtered by
            parameter-name substring ``id``.
        """
        mean_grads = None
        losses = None
        for _ in range(num_samples):
            # import ipdb; ipdb.set_trace()
            if mean_grads is None and losses == None:
                mean_grads, losses = self._grad(name_filter)
            else:
                mean_grad, loss = self._grad(name_filter)
                mean_grads += mean_grad
                losses += loss

        return mean_grads / float(num_samples), losses / float(num_samples)


# ----------------------------------------------------------------------------- #
# Score wrapper class
# ----------------------------------------------------------------------------- #
class ScoreAPI(LossWrapper):
    """Loss wrapper that turns model outputs into scalar rewards.

    Workflow
    --------
    - Sample replies via :meth:`reinforce_loss` to obtain ``(-nll, replies)``.
    - Score each reply with a *frozen* reward model (subclasses implement
      :meth:`reward`).
    - Return the policy loss together with the reward so callers can inspect it.

    Notes
    -----
    The reward model is expected to be *frozen* (no grads) and may be shared
    across instances via the global cache.
    """

    def __init__(
        self,
        prompt: str,
        enc: BatchEncoding,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
        reward_model_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct a score-based loss wrapper.

        Parameters
        ----------
        enc : dict
            Tokenised inputs (e.g., ``tokenizer(prompt_batch, return_tensors='pt')``).
        model : torch.nn.Module
            The *policy* network whose parameters you optimise.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer for the policy model.
        gen_cfg : transformers.GenerationConfig or dict-like
            Generation configuration passed to ``model.generate``.
        reward_model_id : str
            Hugging Face model id for the reward model (sequence classification).
        device : str | torch.device | None, optional
            Where to place the policy and reward model (e.g., "cuda", "cpu").
            If ``None``, inherit :class:`LossWrapper`'s default.
        dtype : torch.dtype | None, optional
            Floating-point precision for the reward model (e.g., ``torch.float16``).
        """
        super().__init__(
            prompt, enc, model, tokenizer, gen_cfg, device=device, dtype=dtype
        )

        if reward_model_id == "Ray2333/GRM-Gemma-2B-rewardmodel-ft":
            # Try to reuse a cached reward model with the same (id, dtype) tuple.
            reward_assets = _get_reward_assets(reward_model_id, dtype=dtype)

            if reward_assets is None:
                # Load a fresh reward model and tokenizer, then cache them.
                # ``device_map`` can be "auto" or a mapping; here we rely on
                # HF's default placement and then push to ``self.device`` below if needed.
                self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self._REWARD_MODEL_ID,
                    torch_dtype=dtype,
                    device_map=self.device,
                )
                self.reward_tokenizer = AutoTokenizer.from_pretrained(
                    self._REWARD_MODEL_ID
                )
                _set_reward_assets(
                    self._REWARD_MODEL_ID,
                    dtype=dtype,
                    model=self.reward_model,
                    tokenizer=self.reward_tokenizer,
                )
            else:
                # Reuse the cached copy; move to the current device if necessary.
                self.reward_model = reward_assets.model.to(self.device)
                self.reward_tokenizer = reward_assets.tokenizer

        elif reward_model_id == "original":
            self.reward_model = Detoxify(reward_model_id)

    # Subclasses *must* supply the scalar reward
    @abstractmethod
    def reward(self, responses) -> torch.tensor: ...

    def loss(self) -> torch.Tensor:
        """Standard REINFORCE loss returning both scalar loss and reward.

        Returns
        -------
        (loss, reward)
            ``loss`` keeps its full autograd graph. ``reward`` is typically a
            detached tensor produced by the reward model (subclass-defined).
        """
        loss, response = self.reinforce_loss()
        reward = self.reward(response)
        return loss, reward


# ----------------------------------------------------------------------------- #
# REINFORCE loss with GRM Gemma 2B
# ----------------------------------------------------------------------------- #
class GRM_Gemma_2B(ScoreAPI):
    """Score with ``Ray2333/GRM-Gemma-2B-rewardmodel-ft`` and minimise -reward.

    The reward model converts each (prompt, response) pair into a scalar; the
    policy objective is the negative of that reward, compatible with vanilla
    gradient descent.
    """

    _REWARD_MODEL_ID: str = "Ray2333/GRM-Gemma-2B-rewardmodel-ft"

    def __init__(
        self,
        prompt: str,
        enc: BatchEncoding,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialise the GRM-based scorer.

        Parameters
        ----------
        enc, model, tokenizer, gen_cfg, device, dtype
            See :class:`ScoreWrapper`.
        """
        super().__init__(
            prompt,
            enc,
            model,
            tokenizer,
            gen_cfg,
            self._REWARD_MODEL_ID,
            device=device,
            dtype=dtype,
        )

    def reward(self, responses) -> torch.tensor:
        """Compute scalar rewards for a batch of decoded responses.

        Steps
        -----
        1. Build chat-formatted strings using the reward model's template.
        2. Tokenize with padding/truncation.
        3. Run the frozen reward model to obtain logits and return as floats.
        """
        # 1) Assemble chat turns for each (prompt, response) pair.
        chats = [
            self.reward_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": self.prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )
            for response in responses
        ]
        # 2) Batch-tokenize.
        inp = self.reward_tokenizer(
            chats, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # 3) Forward pass (no grads) to obtain scalar rewards.
        with torch.no_grad():
            reward = self.reward_model(**inp).logits.squeeze().float()
        return reward


# ----------------------------------------------------------------------------- #
# REINFORCE loss with Detoxify.
# ----------------------------------------------------------------------------- #
class DetoxifyOriginal(ScoreAPI):
    """Score with ``Ray2333/GRM-Gemma-2B-rewardmodel-ft`` and minimise -reward.

    The reward model converts each (prompt, response) pair into a scalar; the
    policy objective is the negative of that reward, compatible with vanilla
    gradient descent.
    """

    _REWARD_MODEL_ID: str = "original"

    def __init__(
        self,
        prompt: str,
        enc: BatchEncoding,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialise the GRM-based scorer.

        Parameters
        ----------
        enc, model, tokenizer, gen_cfg, device, dtype
            See :class:`ScoreWrapper`.
        """
        super().__init__(
            prompt,
            enc,
            model,
            tokenizer,
            gen_cfg,
            self._REWARD_MODEL_ID,
            device=device,
            dtype=dtype,
        )

    def reward(self, responses) -> torch.tensor:
        """Compute scalar rewards for a batch of decoded responses."""
        # Forward pass (no grads) to obtain scalar rewards.
        with torch.no_grad():
            reward = self.reward_model.predict(responses)
        toxicity_reward = list(reward.values())[0]
        return torch.tensor(toxicity_reward, device=self.device)


# ----------------------------------------------------------------------------- #
# Reinforce loss with Semgrep
# ----------------------------------------------------------------------------- #
class Semgrep(LossWrapper):
    """Score responses using Semgrep and minimize -reward.

    Each (prompt, response) pair is evaluated with Semgrep rules to produce
    a scalar reward. The policy objective is the negative of that reward,
    compatible with standard gradient descent / REINFORCE-style optimization.
    """

    def __init__(
        self,
        prompt: str,
        enc: BatchEncoding,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
        *,
        rule: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the Semgrep scorer.

        Parameters
        ----------
        prompt : str
            The input prompt text.
        enc : BatchEncoding
            Tokenized encoding of the prompt.
        model : torch.nn.Module
            The policy model used for generating responses.
        tokenizer : Any
            Tokenizer associated with the model.
        gen_cfg : Any
            Generation configuration for producing responses.
        rule : dict
            Semgrep rule to apply for scoring responses.
        device : str or torch.device
            Device to store reward tensor on.
        dtype : torch.dtype
            Data type of reward tensor.
        """
        super().__init__(
            prompt, enc, model, tokenizer, gen_cfg, device=device, dtype=dtype
        )

        # Crate temp folder
        self.temp_dir = tempfile.mkdtemp(prefix="semgrep_")

        # Create path for semgrep instructions file or folder
        if rule is None and SEMGREP_INSTRUCTIONS_PATH.exists():
            self.rule_path = str(SEMGREP_INSTRUCTIONS_PATH)
        elif rule is not None and os.path.exists(rule):
            self.rule_path = rule
        else:
            raise ValueError("Semgrep rules not found")

    def __enter__(self):
        """Support for context manager usage (with-statement)."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Clean up temporary folder and all files when exiting context."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def reward(self, responses: list[str]) -> torch.tensor:
        """Compute a reward tensor for a list of generated responses.

        Each response is saved to a temporary Python file and analyzed
        using Semgrep with the configured rule. The reward is simply
        the number of Semgrep matches in the response.

        Parameters
        ----------
        responses : list[str]
            List of generated Python code responses to score.

        Returns
        -------
        torch.Tensor
            Tensor of shape (len(responses),) containing the number of matches
            as floats, stored on the configured device.
        """
        num_matches_list = []

        for response in responses:
            # Save the response code to a temporary Python file
            code_file = os.path.join(self.temp_dir, "temp_code.py")
            with open(code_file, "w") as f:
                f.write(response)

            code_file_path = clean_and_save(response)

            # Run Semgrep on the generated files
            for code_file in code_file_path:
                num_matches = 0

                proc = subprocess.run(
                    ["semgrep", "--config", self.rule_path, "--json", code_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                res = json.loads(proc.stdout)
                output = res["results"]

                # Remove temporary Python file immediately
                os.remove(code_file)

                # Raise an error if the Semgrep rule was invalid
                if "errors" in output and len(output["errors"]) > 0:
                    for err in output.get("errors", []):
                        if err.get("type") == "SemgrepError":
                            raise RuntimeError("invalid rule")

                # Count the number of matches found by Semgrep
                num_matches += -len(output)
            num_matches_list.append(num_matches)

        # Convert the list of counts into a float tensor on the configured device
        return torch.tensor(num_matches_list, dtype=torch.float, device=self.device)

    def loss(self) -> torch.Tensor:
        """Compute standard REINFORCE-style loss.

        The loss is computed using the policy's reinforce_loss method,
        and the reward is obtained from Semgrep analysis of the responses.

        Returns
        -------
        (loss, reward) : tuple[torch.Tensor, torch.Tensor]
            loss : scalar tensor with full autograd graph
            reward : detached tensor produced by Semgrep reward model
        """
        # Compute REINFORCE loss and get generated responses
        loss, response = self.reinforce_loss()

        # Compute reward for generated responses using Semgrep
        reward = self.reward(response)

        return loss, reward


# ----------------------------------------------------------------------------- #
# ASDL-compatible score-based loss modules
# ----------------------------------------------------------------------------- #
class ScoreAPILoss(torch.nn.Module):
    """Score-based loss wrapper compatible with ASDL loss_fn usage.

    This class exposes a `set_context()` method to inject the prompt/model inputs
    before `forward()` is called by ASDL. The `forward()` method returns a scalar
    tensor suitable for `loss.backward()`.
    """

    def __init__(
        self,
        reward_model_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reward_model_id = reward_model_id
        self.device = device
        self.dtype = dtype
        self.reduction = reduction

        self._context_set = False
        self._last_reward: torch.Tensor | None = None

        self._model: torch.nn.Module | None = None
        self._tokenizer = None
        self._gen_cfg = None
        self._enc = None
        self._prompt = ""
        self._prompt_len = 0
        self._pad_id = None

        self._reward_model = None
        self._reward_tokenizer = None

    def _ensure_reward_assets(self) -> None:
        if self._reward_model is None:
            if self.reward_model_id == "original":
                self._reward_model = Detoxify(self.reward_model_id)
                return

            reward_assets = _get_reward_assets(self.reward_model_id, dtype=self.dtype)
            if reward_assets is None:
                device_map = self.device
                if isinstance(device_map, torch.device):
                    device_map = str(device_map)
                kwargs = {"torch_dtype": self.dtype}
                if device_map is not None:
                    kwargs["device_map"] = device_map
                reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self.reward_model_id,
                    **kwargs,
                )
                reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_id)
                _set_reward_assets(
                    self.reward_model_id,
                    dtype=self.dtype,
                    model=reward_model,
                    tokenizer=reward_tokenizer,
                )
                reward_assets = _get_reward_assets(
                    self.reward_model_id, dtype=self.dtype
                )
            self._reward_model = reward_assets.model
            self._reward_tokenizer = reward_assets.tokenizer

        if self.device is not None and hasattr(self._reward_model, "to"):
            self._reward_model = self._reward_model.to(self.device)

    def set_context(
        self,
        prompt: str,
        enc,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
    ) -> None:
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            self._model = model
        else:
            self._model = model.to(self.device, dtype=self.dtype)

        self._tokenizer = tokenizer
        self._gen_cfg = gen_cfg
        self._pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )

        if getattr(model, "is_loaded_in_8bit", True) or getattr(
            model, "is_loaded_in_4bit", True
        ):
            self.device = model.get_input_embeddings().weight.device
        elif self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._enc = {k: v.to(self.device) for k, v in enc.items()}
        self._prompt = prompt
        self._prompt_len = self._enc["input_ids"].shape[1]
        self._ensure_reward_assets()
        self._context_set = True

    @property
    def last_reward(self) -> torch.Tensor | None:
        """Return the most recent reward tensor (detached)."""
        return self._last_reward

    def _reinforce_loss(self) -> tuple[torch.Tensor, list[str]]:
        if self._model is None or self._enc is None:
            raise RuntimeError("Must call set_context() before forward().")

        with torch.no_grad():
            generated_ids = self._model.generate(
                **self._enc,
                generation_config=self._gen_cfg,
                return_dict_in_generate=False,
                output_scores=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        labels = torch.full_like(generated_ids, -100)
        if self._prompt_len > 0:
            labels[:, self._prompt_len :] = generated_ids[:, self._prompt_len :]
        else:
            labels.copy_(generated_ids)
        labels.masked_fill_(generated_ids.eq(self._pad_id), -100)
        attn = generated_ids.ne(self._pad_id).to(dtype=torch.long)

        fw = self._model(
            input_ids=generated_ids,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        reply_ids = generated_ids[:, self._prompt_len :].detach()
        replies = self._tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        del reply_ids, attn, labels
        torch.cuda.empty_cache()

        return -fw.loss, replies

    def _reward(self, responses: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if not self._context_set:
            raise RuntimeError("Must call set_context() before forward().")

        loss, responses = self._reinforce_loss()
        reward = self._reward(responses)
        self._last_reward = reward.detach()

        weighted = loss * reward
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        return weighted.mean()


class GRM_Gemma_2BLoss(ScoreAPILoss):
    """ASDL-compatible loss for ``Ray2333/GRM-Gemma-2B-rewardmodel-ft``."""

    _REWARD_MODEL_ID: str = "Ray2333/GRM-Gemma-2B-rewardmodel-ft"

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            self._REWARD_MODEL_ID,
            device=device,
            dtype=dtype,
            reduction=reduction,
        )

    def _reward(self, responses: Sequence[str]) -> torch.Tensor:
        if self._reward_model is None or self._reward_tokenizer is None:
            raise RuntimeError("Reward model not initialized.")

        chats = [
            self._reward_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": self._prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )
            for response in responses
        ]
        inp = self._reward_tokenizer(
            chats, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            reward = self._reward_model(**inp).logits.squeeze().float()
        return reward


class DetoxifyOriginalLoss(ScoreAPILoss):
    """ASDL-compatible loss for Detoxify ``original``."""

    _REWARD_MODEL_ID: str = "original"

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            self._REWARD_MODEL_ID,
            device=device,
            dtype=dtype,
            reduction=reduction,
        )

    def _reward(self, responses: Sequence[str]) -> torch.Tensor:
        if self._reward_model is None:
            raise RuntimeError("Reward model not initialized.")

        with torch.no_grad():
            reward = self._reward_model.predict(list(responses))
        toxicity_reward = list(reward.values())[0]
        return torch.tensor(toxicity_reward, device=self.device)


class SemgrepLoss(nn.Module):
    """Semgrep-based loss wrapper compatible with ASDL loss_fn usage.

    This class exposes a `set_context()` method to inject the prompt/model inputs
    before `forward()` is called by ASDL. The `forward()` method returns a scalar
    tensor suitable for `loss.backward()`.
    """

    def __init__(
        self,
        rule: Optional[str] = None,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        reduction: str = "mean",
        timeout: int = 10,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.reduction = reduction
        self.timeout = timeout

        self._context_set = False
        self._temp_dir = tempfile.mkdtemp(prefix="semgrep_")
        self._last_reward: Optional[torch.Tensor] = None
        self.rule_path = self._resolve_rule_path(rule)

        self._model: Optional[torch.nn.Module] = None
        self._tokenizer = None
        self._gen_cfg = None
        self._enc = None
        self._prompt_len = 0
        self._pad_id = None

    def _resolve_rule_path(self, rule: Optional[str]) -> str:
        """Resolve the Semgrep rule path from an explicit rule or default folder."""
        if rule is None:
            if SEMGREP_INSTRUCTIONS_PATH.exists():
                return str(SEMGREP_INSTRUCTIONS_PATH)
            raise ValueError(
                f"Semgrep rules not found at default path: {SEMGREP_INSTRUCTIONS_PATH}"
            )
        if not os.path.exists(rule):
            raise ValueError(f"Semgrep rules not found at: {rule}")
        return rule

    def set_context(
        self,
        prompt: str,
        enc,
        model: torch.nn.Module,
        tokenizer,
        gen_cfg,
    ) -> None:
        """Set the per-step context required for `forward()`."""
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            self._model = model
        else:
            self._model = model.to(self.device, dtype=self.dtype)

        self._tokenizer = tokenizer
        self._gen_cfg = gen_cfg
        self._pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )

        if getattr(model, "is_loaded_in_8bit", True) or getattr(
            model, "is_loaded_in_4bit", True
        ):
            self.device = model.get_input_embeddings().weight.device
        elif self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._enc = {k: v.to(self.device) for k, v in enc.items()}
        self._prompt_len = self._enc["input_ids"].shape[1]
        self._context_set = True

    @property
    def last_reward(self) -> Optional[torch.Tensor]:
        """Return the most recent reward tensor (detached)."""
        return self._last_reward

    def _reinforce_loss(self) -> tuple[torch.Tensor, List[str]]:
        """Compute REINFORCE loss and decode replies."""
        if self._model is None or self._enc is None:
            raise RuntimeError("Must call set_context() before forward().")

        with torch.no_grad():
            generated_ids = self._model.generate(
                **self._enc,
                generation_config=self._gen_cfg,
                return_dict_in_generate=False,
                output_scores=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        labels = torch.full_like(generated_ids, -100)
        if self._prompt_len > 0:
            labels[:, self._prompt_len :] = generated_ids[:, self._prompt_len :]
        else:
            labels.copy_(generated_ids)
        labels.masked_fill_(generated_ids.eq(self._pad_id), -100)
        attn = generated_ids.ne(self._pad_id).to(dtype=torch.long)

        fw = self._model(
            input_ids=generated_ids,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        reply_ids = generated_ids[:, self._prompt_len :].detach()
        replies = self._tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        del reply_ids, attn, labels
        torch.cuda.empty_cache()

        return -fw.loss, replies

    def _run_semgrep(self, code_path: str) -> int:
        """Run Semgrep on a single file and return the match count."""
        proc = subprocess.run(
            ["semgrep", "--config", self.rule_path, "--json", code_path],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        try:
            res = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Semgrep output is not valid JSON: {proc.stdout}"
            ) from exc

        results = res.get("results", [])
        errors = res.get("errors", [])
        for err in errors:
            if err.get("type") == "SemgrepError":
                raise RuntimeError("invalid rule")

        return len(results)

    def _reward(self, responses: Iterable[str]) -> torch.Tensor:
        """Compute a reward tensor for a list of generated responses."""
        matches: List[int] = []
        for response in responses:
            code_paths = clean_and_save(response)
            num_matches = 0
            for path in code_paths:
                num_matches += -self._run_semgrep(path)
                self._safe_unlink(path)
            matches.append(num_matches)
        return torch.tensor(matches, dtype=torch.float, device=self.device)

    def _safe_unlink(self, path: str) -> None:
        """Best-effort cleanup for temporary files."""
        try:
            os.remove(path)
        except FileNotFoundError:
            return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute a scalar Semgrep loss suitable for backpropagation."""
        if not self._context_set:
            raise RuntimeError("Must call set_context() before forward().")

        loss, responses = self._reinforce_loss()
        reward = self._reward(responses)
        self._last_reward = reward.detach()

        weighted = loss * reward
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        return weighted.mean()

    def close(self) -> None:
        """Clean up temporary resources."""
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            try:
                for name in os.listdir(self._temp_dir):
                    self._safe_unlink(os.path.join(self._temp_dir, name))
                os.rmdir(self._temp_dir)
            except OSError:
                return

    def __del__(self) -> None:
        self.close()
