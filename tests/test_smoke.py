"""CPU smoke tests for the influence pipeline on a tiny random Llama model.

They need no GPU and no downloads, and run in a few seconds:

    pip install -e . pytest
    pytest tests/

They check that every K-FAC code path runs end to end and returns finite
gradients (including half-precision models, which used to crash on the cached
and generic paths), and that saving and reloading state round-trips.
"""
import contextlib
import io

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("asdl")

from asdl.precondition import PreconditioningConfig  # noqa: E402

from IF.Influence.Influence import Influences  # noqa: E402
from IF.KFAC.small.advance_small_KFAC_gradient_maker import (  # noqa: E402
    AdvancedSmallKfacGradientMaker,
)

TRAIN = ["def f(x): return x+1", "import os", "print('hi')", "for i in range(3): pass"]
EVAL = ["import os\nos.system(input())"]
REGEX = r"model\.layers\.(0|2)\.mlp\.(up_proj|down_proj)\.weight$"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 ()=:.,_+-*/'\"[]{}")
    vocab = {c: i for i, c in enumerate(["<unk>", "<eos>"] + chars)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")


def _run(tokenizer, dtype, **gm_overrides):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg).to(dtype).eval()
    config = PreconditioningConfig(
        data_size=1, damping=0.1, ema_decay=-1, ignore_modules=[nn.Embedding, nn.LayerNorm]
    )
    gm_kwargs = dict(
        use_grad_gram=True, use_woodbury=False, small_matrices=False, short_side=True,
        name_pattern="default", proj_names=("up_proj", "down_proj"), normalize=True,
        use_amp=True, rel_eps=0.1, min_damping=1e-3,
    )
    gm_kwargs.update(gm_overrides)
    with contextlib.redirect_stdout(io.StringIO()):  # the library prints a lot
        gm = AdvancedSmallKfacGradientMaker(model, config, **gm_kwargs)
        inf = Influences(gm, tokenizer, TRAIN, EVAL, device=torch.device("cpu"), dtype="float32")
        grads = inf.candidateInfluence()
        inf.layer_influence_score(prompt_text=TRAIN[1])
    return gm, inf, grads


@pytest.mark.parametrize(
    "dtype, overrides, n_blocks",
    [
        (torch.float32, {}, 8),                              # paper configuration
        (torch.bfloat16, {}, 8),
        (torch.float32, {"use_woodbury": True}, 8),
        (torch.bfloat16, {"use_small_cache": True}, 8),
        (torch.bfloat16, {"name_pattern": REGEX}, 4),        # generic path
        (torch.bfloat16, {"name_pattern": REGEX, "short_side": False}, 4),
    ],
    ids=["default-fp32", "default-bf16", "woodbury", "small-cache-bf16",
         "generic-bf16", "generic-two-sided-bf16"],
)
def test_preconditioning_paths(tokenizer, dtype, overrides, n_blocks):
    _, _, grads = _run(tokenizer, dtype, **overrides)
    assert len(grads) == n_blocks
    for name, g in grads.items():
        assert torch.isfinite(g.float()).all(), name
        assert g.float().abs().sum() > 0, name


def test_save_load_round_trip(tokenizer, tmp_path):
    gm, inf, grads = _run(tokenizer, torch.float32)
    prefix = str(tmp_path / "state")
    with contextlib.redirect_stdout(io.StringIO()):
        inf.save(prefix)
        gm2, inf2, _ = _run(tokenizer, torch.float32)
        gm2.clear_curvature_cache(force_gc=False)
        gm2.load_state(prefix + "_gm.pt")
        inf2.load_state(prefix + "_influence.pt")

    for k, v in gm.A_u_dict.items():
        if v is not None:
            assert torch.equal(v, gm2.A_u_dict[k])
    for k, v in grads.items():
        assert torch.equal(v, inf2.conditioned_grads[k])


def test_generic_state_round_trip(tokenizer, tmp_path):
    gm, _, _ = _run(tokenizer, torch.float32, name_pattern=REGEX)
    path = str(tmp_path / "gm.pt")
    with contextlib.redirect_stdout(io.StringIO()):
        gm.save_state(path)
        gm2, _, _ = _run(tokenizer, torch.float32, name_pattern=REGEX)
        gm2.clear_curvature_cache(force_gc=False)
        gm2.load_state(path)
    assert set(gm2._param_A) == set(gm._param_A) and len(gm._param_A) == 4
    for k in gm._param_A:
        assert torch.equal(gm._param_A[k], gm2._param_A[k])
    assert dict(gm2._param_counts) == dict(gm._param_counts)
