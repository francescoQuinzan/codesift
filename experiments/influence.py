# -----------------------------------------------------------------------------
# Influence-norm computation for sampled model answers.
#
# Single entry point replacing the four per-model scripts
# (deepseek-coder-6.7b-instruct, granite-3b-code-instruct-2k,
# OpenCodeInterpreter-CL-13B, Qwen2.5-Coder-7B-Instruct).
#
# For one prompt, samples --n_trials answers, computes the K-FAC-conditioned
# influence gradients of each answer, and stores their L0.5/L1/L1.5/L2/LInf
# norms together with uncertainty baselines (NLL, perplexity, mean token
# entropy, length) of the same sampled completion.
#
# Example (select the GPU with the environment variable):
#   CUDA_VISIBLE_DEVICES=1 python influence.py --model qwen \
#       --evalPrompt "Write a function ..." --output_file results/qwen_if.pkl
#
# Output:
#   out[evalPrompt][answer] = {"L0.5": [...], "L1": [...], "L1.5": [...],
#                              "L2": [...], "LInf": [...],
#                              "nll": [...], "perplexity": [...],
#                              "mean_entropy": [...], "length": [...]}
#   (one list entry per time that answer was sampled; all lists are aligned)
#
# The uncertainty metrics are computed on the exact tokens the model generated
# (the full raw completion, before --extract post-processing), teacher-forced,
# with the prompt tokens excluded.
# -----------------------------------------------------------------------------

import argparse
import dataclasses
import gc
import math
import os
import pickle
import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList, set_seed)

from IF.KFAC.small.advance_small_KFAC_gradient_maker import AdvancedSmallKfacGradientMaker
from asdl.precondition import PreconditioningConfig
from IF.Influence.Influence import Influences
from IF import PROJECT_ROOT

device = torch.device("cuda:0")


# -----------------------------------------------------------------------------
# Model presets (reproduce the behaviour of the four original scripts)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPreset:
    model_id: str
    torch_dtype: str = "auto"                  # "auto" | "float32" | "bfloat16" | "float16"
    device_map: str = "cuda:0"
    trust_remote_code: bool = False
    disable_default_system_prompt: bool = False
    add_pad_token: bool = False                # add [PAD] + resize embeddings if missing
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_k: Optional[int] = None                # None -> model's generation_config default
    top_p: float = 0.95
    clamp_logits: bool = False                 # replace inf/nan logits before sampling
    strip_prompt: bool = True                  # drop prompt tokens from the decoded output
    fix_bpe_artifacts: bool = False            # replace stray "Ġ"/"Ċ" in decoded text
    extract: str = "code_block"                # "answer_split" | "code_block" | "none"


MODEL_PRESETS = {
    "deepseek": ModelPreset(
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        disable_default_system_prompt=True,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.92,
        fix_bpe_artifacts=True,
        extract="none",
    ),
    "granite": ModelPreset(
        model_id="ibm-granite/granite-3b-code-instruct-2k",
        torch_dtype="float32",
        max_new_tokens=256,
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        strip_prompt=False,
        extract="answer_split",
    ),
    "opencodeinterpreter": ModelPreset(
        model_id="m-a-p/OpenCodeInterpreter-CL-13B",
        torch_dtype="float16",
        device_map="auto",
        add_pad_token=True,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        clamp_logits=True,
        extract="code_block",
    ),
    "qwen": ModelPreset(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype="auto",
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.95,
        extract="code_block",
    ),
}

NORMS = {"L0.5": 0.5, "L1": 1.0, "L1.5": 1.5, "L2": 2.0, "LInf": float("inf")}
UNCERTAINTY = ("nll", "perplexity", "mean_entropy", "length")


# -----------------------------------------------------------------------------
# Input parser
# -----------------------------------------------------------------------------

def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sample answers to a prompt and compute the norms of their K-FAC-conditioned "
                    "influence gradients. Select the GPU with CUDA_VISIBLE_DEVICES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("I/O")
    g.add_argument("--evalPrompt", type=str, required=True,
                   help="Evaluation prompt (use quotes; newlines are allowed).")
    g.add_argument("--output_file", type=str, required=True,
                   help="Output pickle file (results for this prompt are merged into it).")
    g.add_argument("--workdir", type=str, default=str(PROJECT_ROOT),
                   help="Directory to chdir into before loading data (relative paths resolve here).")
    g.add_argument("--train_data", type=str, default="datasets/clean_instructions.pkl",
                   help="Pickle with the training examples used for the K-FAC / influence computation.")
    g.add_argument("--n_train", type=int, default=99,
                   help="Number of training examples to use (first N of --train_data).")
    g.add_argument("--n_trials", type=int, default=10,
                   help="Number of answers to sample for the prompt.")
    g.add_argument("--print_answers", action="store_true",
                   help="Print every sampled answer.")

    g = p.add_argument_group("Model (defaults come from the model preset)")
    g.add_argument("--model", type=str, required=True,
                   help=f"Preset name ({', '.join(MODEL_PRESETS)}) or a Hugging Face model id "
                        f"(an HF id uses the generic ModelPreset defaults unless overridden).")
    g.add_argument("--torch_dtype", type=str, default=None,
                   choices=["auto", "float32", "bfloat16", "float16"])
    g.add_argument("--device_map", type=str, default=None,
                   help="HF device_map, e.g. cuda:0 or auto (splits the model over all visible GPUs).")
    g.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--load_in_4bit", action="store_true",
                   help="Load the model in 4-bit NF4 via bitsandbytes.")

    g = p.add_argument_group("Generation (defaults come from the model preset)")
    g.add_argument("--system_prompt", type=str, default=None,
                   help="System prompt; pass an empty string to disable.")
    g.add_argument("--max_new_tokens", type=int, default=None)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top_k", type=int, default=None, help="Use -1 to leave unset.")
    g.add_argument("--top_p", type=float, default=None)
    g.add_argument("--greedy", action="store_true",
                   help="Greedy decoding (do_sample=False); all trials then give the same answer.")
    g.add_argument("--clamp_logits", action=argparse.BooleanOptionalAction, default=None,
                   help="Replace inf/nan logits before sampling (useful in float16).")
    g.add_argument("--strip_prompt", action=argparse.BooleanOptionalAction, default=None,
                   help="Drop the prompt tokens from the decoded output.")
    g.add_argument("--extract", type=str, default=None,
                   choices=["answer_split", "code_block", "none"],
                   help="Post-processing of the generated text: split on 'Answer:', take the "
                        "first ``` code block (falls back to the raw text), or keep the raw text.")
    g.add_argument("--seed", type=int, default=0,
                   help="Random seed for generation; use -1 for a non-deterministic run.")

    g = p.add_argument_group("K-FAC / influence")
    g.add_argument("--proj_names", type=_csv, default=["up_proj", "down_proj"],
                   help="Projections the gradient maker targets.")
    g.add_argument("--keep_layers", type=str, default="all",
                   help="'all', 'quarter' (the model's quarter-point layers), or comma-separated "
                        "layer indices. Everything else is added to ignore_modules.")
    g.add_argument("--use_grad_gram", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--use_woodbury", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--short_side", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--rel_eps", type=float, default=0.1,
                   help="Relative epsilon for the KFAC computation.")
    g.add_argument("--min_damping", type=float, default=1e-3,
                   help="KFAC damping floor.")
    g.add_argument("--damping", type=float, default=0.1,
                   help="PreconditioningConfig damping.")
    g.add_argument("--epsilon", type=float, default=1e-3,
                   help="Epsilon parameter for the influence computation.")
    g.add_argument("--if_dtype", type=str, default="float16",
                   help="dtype passed to Influences().")
    g.add_argument("--fail_fast", action="store_true",
                   help="Raise on the first failing answer instead of logging it and continuing.")

    args = p.parse_args(argv)

    preset = MODEL_PRESETS.get(args.model) or ModelPreset(model_id=args.model)
    overrides = {k: getattr(args, k) for k in (
        "torch_dtype", "device_map", "trust_remote_code", "system_prompt", "max_new_tokens",
        "temperature", "top_k", "top_p", "clamp_logits", "strip_prompt", "extract",
    ) if getattr(args, k) is not None}
    if overrides.get("top_k") == -1:
        overrides["top_k"] = None
    if overrides.get("system_prompt") == "":
        overrides["system_prompt"] = None
    args.preset = dataclasses.replace(preset, **overrides)
    return args


# -----------------------------------------------------------------------------
# Model loading and generation
# -----------------------------------------------------------------------------

def load_model(preset, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(
        preset.model_id, trust_remote_code=preset.trust_remote_code)
    if preset.disable_default_system_prompt:
        tokenizer.use_default_system_prompt = False

    kwargs = dict(trust_remote_code=preset.trust_remote_code, device_map=preset.device_map)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = (preset.torch_dtype if preset.torch_dtype == "auto"
                                 else getattr(torch, preset.torch_dtype))

    model = AutoModelForCausalLM.from_pretrained(preset.model_id, **kwargs)

    if preset.add_pad_token and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer


class ClampInfNanProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = scores.to(torch.float32)  # upcast before softmax
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


def build_inputs(prompt, tokenizer, preset, device_):
    messages = []
    if preset.system_prompt:
        messages.append({"role": "system", "content": preset.system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    except Exception as e:
        print(f"[warn] chat template failed ({e}); falling back to [INST] formatting.")
        enc = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt")
    return {k: v.to(device_) for k, v in enc.items()}


def postprocess(text, preset):
    if preset.fix_bpe_artifacts:
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
    if preset.extract == "answer_split":
        return text.split("Answer:", 1)[1].strip() if "Answer:" in text else text.strip()
    if preset.extract == "code_block":
        m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        print("[warn] no ``` code block found in the generation; using the raw text.")
    return text.strip()


@torch.inference_mode()
def completion_metrics(seq_ids, prompt_len, model, clamp_logits=False):
    """Uncertainty baselines for one sampled completion.

    `seq_ids` is the (1, L) output of model.generate (prompt + completion).
    One teacher-forced forward pass scores the completion tokens only:
      nll           total negative log-likelihood of the completion (nats)
      perplexity    exp(nll / length)
      mean_entropy  mean entropy of the next-token distribution (nats)
      length        number of completion tokens
    """
    T = seq_ids.shape[1] - prompt_len
    if T <= 0:
        return {"nll": 0.0, "perplexity": 1.0, "mean_entropy": 0.0, "length": 0}

    # Logits at positions prompt_len-1 .. L-2 predict the completion tokens.
    logits = model(seq_ids).logits[0, prompt_len - 1:-1].float()     # (T, V)
    if clamp_logits:
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    targets = seq_ids[0, prompt_len:].to(logits.device)

    nll = F.cross_entropy(logits, targets, reduction="sum").item()
    entropy = torch.distributions.Categorical(logits=logits).entropy()
    return {"nll": nll, "perplexity": math.exp(nll / T),
            "mean_entropy": entropy.mean().item(), "length": T}


def sample_answers(prompt, model, tokenizer, preset, n, greedy=False):
    """Returns a list of (answer_text, uncertainty_metrics) pairs."""
    inputs = build_inputs(prompt, tokenizer, preset, model.device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=preset.max_new_tokens,
        do_sample=not greedy,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    if not greedy:
        for k in ("temperature", "top_k", "top_p"):
            if getattr(preset, k) is not None:
                gen_kwargs[k] = getattr(preset, k)
    if preset.clamp_logits:
        gen_kwargs["logits_processor"] = LogitsProcessorList([ClampInfNanProcessor()])

    samples = []
    for _ in range(n):
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        metrics = completion_metrics(output_ids, prompt_len, model, preset.clamp_logits)
        if preset.strip_prompt:
            output_ids = output_ids[:, prompt_len:]
        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        samples.append((postprocess(text, preset), metrics))
    return samples


# -----------------------------------------------------------------------------
# K-FAC setup and norms
# -----------------------------------------------------------------------------

def resolve_keep_layers(model, spec):
    if spec == "all":
        return None
    n_layers = model.config.num_hidden_layers
    if spec == "quarter":
        return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4})
    layers = sorted({int(x) for x in _csv(spec)})
    bad = [i for i in layers if i >= n_layers]
    if bad:
        raise ValueError(f"--keep_layers {bad} out of range for num_hidden_layers={n_layers}")
    return layers


def build_ignore(model, keep_layers, keep_proj):
    """[Embedding, LayerNorm], plus (if keep_layers is given) every Linear that
    is not one of `keep_proj` in `keep_layers`."""
    ignore = [nn.Embedding, nn.LayerNorm]
    if keep_layers is None:
        return ignore
    keep_layers, keep_proj = set(keep_layers), tuple(keep_proj)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not (m and int(m.group(1)) in keep_layers and name.endswith(keep_proj)):
            ignore.append(mod)
    return ignore


def lp_norms(blocks):
    """Lp norms of the concatenation of `blocks`, computed block by block in
    float32 (identical to torch.norm(torch.cat([b.flatten() ...]), p) but without
    building the concatenated vector, and without float16 overflow)."""
    sums = {name: 0.0 for name, p in NORMS.items() if p != float("inf")}
    max_abs = 0.0
    for b in blocks:
        a = b.detach().to(torch.float32).abs().flatten()
        for name, p in NORMS.items():
            if p == float("inf"):
                max_abs = max(max_abs, a.max().item())
            elif p == 1.0:
                sums[name] += a.sum().item()
            elif p == 2.0:
                sums[name] += a.square().sum().item()
            else:
                sums[name] += a.pow(p).sum().item()
    out = {name: s ** (1.0 / NORMS[name]) for name, s in sums.items()}
    out["LInf"] = max_abs
    return {name: out[name] for name in NORMS}


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    preset = args.preset
    os.chdir(args.workdir)

    if args.seed >= 0:
        set_seed(args.seed)

    model, tokenizer = load_model(preset, load_in_4bit=args.load_in_4bit)

    with open(args.train_data, "rb") as f:
        train_data = pickle.load(f)[: args.n_train]

    samples = sample_answers(args.evalPrompt, model, tokenizer, preset,
                             n=args.n_trials, greedy=args.greedy)
    answers = [a for a, _ in samples]
    print(f"Sampled {len(answers)} answers ({len(set(answers))} distinct).")
    if args.print_answers:
        for i, a in enumerate(answers):
            print(f"---- answer {i} ----\n{a}")
    free_cuda()

    keep_layers = resolve_keep_layers(model, args.keep_layers)
    config = PreconditioningConfig(
        data_size=1,
        damping=args.damping,
        ema_decay=-1,
        ignore_modules=build_ignore(model, keep_layers, args.proj_names),
    )
    gm = AdvancedSmallKfacGradientMaker(
        model,
        config,
        use_grad_gram=args.use_grad_gram,
        use_woodbury=args.use_woodbury,
        small_matrices=False,
        short_side=args.short_side,
        name_pattern="default",
        proj_names=tuple(args.proj_names),
        normalize=True,
        use_amp=True,
        rel_eps=args.rel_eps,
        min_damping=args.min_damping,
    )

    res, n_failed = {}, 0
    for i, (answer, unc) in enumerate(samples):
        if_calc = None
        try:
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            if_calc = Influences(
                gradient_maker=gm,
                tokenizer=tokenizer,
                train_ds=train_data,
                test_ds=[answer],
                device=device,
                dtype=args.if_dtype,
                epsilon=args.epsilon,
            )
            if_calc.candidateInfluence()
            if_calc.layer_influence_score(prompt_text=answer)

            norms = lp_norms(if_calc.conditioned_grads.values())
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            # Influence norms and uncertainty are appended together, so the
            # lists stay aligned (a failed answer adds to neither).
            entry = res.setdefault(answer, {k: [] for k in (*NORMS, *UNCERTAINTY)})
            for name, val in {**norms, **unc}.items():
                entry[name].append(val)

            print(f"[{i + 1}/{len(answers)}] {elapsed:.1f}s  "
                  + "  ".join(f"{k}={v:.4g}" for k, v in {**norms, **unc}.items()))

        except Exception:
            if args.fail_fast:
                raise
            n_failed += 1
            print(f"[error] answer {i + 1}/{len(answers)} failed:")
            traceback.print_exc()
        finally:
            del if_calc
            free_cuda()

    if n_failed:
        print(f"[warn] {n_failed}/{len(answers)} answers failed and are missing from the results.")

    # -------------------------------------------------------------------------
    # Save results (merge into existing file, write atomically)
    # -------------------------------------------------------------------------

    filename = args.output_file
    out = {}
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            out = pickle.load(f)
    out[args.evalPrompt] = res

    tmp = filename + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, filename)
    print(f"Saved results to {os.path.abspath(filename)}")


if __name__ == "__main__":
    main()