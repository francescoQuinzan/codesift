# -----------------------------------------------------------------------------
# K-FAC approximation-ladder ablation for influence functions.
#
# Single entry point replacing the three per-model scripts
# (deepseek-coder-6.7b-instruct, granite-3b-code-instruct-2k,
# Qwen2.5-Coder-7B-Instruct).
#
# Example (select the GPU with the environment variable):
#   CUDA_VISIBLE_DEVICES=1 python ablation.py --model qwen \
#       --evalPrompt "Write a function ..." --output_file results/qwen.pkl
#
# Per-model defaults (dtype, sampling, system prompt, answer post-processing)
# live in MODEL_PRESETS; any of them can be overridden from the command line.
# -----------------------------------------------------------------------------

import argparse
import dataclasses
import gc
import os
import pickle
import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from IF.KFAC.small.advance_small_KFAC_gradient_maker import AdvancedSmallKfacGradientMaker
from asdl.precondition import PreconditioningConfig
from IF.Influence.Influence import Influences
from IF import PROJECT_ROOT

device = torch.device("cuda:0")


# -----------------------------------------------------------------------------
# Model presets (reproduce the behaviour of the three original scripts)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPreset:
    model_id: str
    torch_dtype: str = "auto"                  # "auto" | "float32" | "bfloat16" | "float16"
    trust_remote_code: bool = False
    disable_default_system_prompt: bool = False
    system_prompt: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_k: Optional[int] = 50                  # None -> model's generation_config default
    top_p: float = 0.95
    strip_prompt: bool = False                 # drop prompt tokens from the decoded output
    extract: str = "answer_split"              # "answer_split" | "code_block" | "none"


MODEL_PRESETS = {
    "deepseek": ModelPreset(
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        disable_default_system_prompt=True,
    ),
    "granite": ModelPreset(
        model_id="ibm-granite/granite-3b-code-instruct-2k",
        torch_dtype="float32",
    ),
    "qwen": ModelPreset(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype="auto",
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        max_new_tokens=512,
        temperature=0.3,
        top_k=None,
        strip_prompt=True,
        extract="code_block",
    ),
}

# -----------------------------------------------------------------------------
# K-FAC approximation ladder (A0-A3, plus off-ladder A2.5a/A2.5b)
#
# A3 is the default baseline. Every other row's conditioned gradients are
# compared against the baseline's, block by block, matched by name.
#
# A0-A3 are a cumulative ladder: only use_grad_gram / use_woodbury / proj_names
# change; short_side and min_damping stay fixed at True / --min_damping.
# A2.5a/A2.5b are an off-ladder reference pair that force the generic code
# path (use_grad_gram=True, use_woodbury has no effect there) and therefore
# need a regex name_pattern (which also restricts the layers, since
# set_target_layers() is silently ignored whenever name_pattern is a regex)
# and a min_damping floor (--offladder_min_damping, default 1e-5).
#
# Every row targets the same layers (--keep_layers, default: the model's
# quarter points); the ignore list is rebuilt per row so that each row's
# proj_names (e.g. q/k/v for A0) are actually preconditioned.
# -----------------------------------------------------------------------------

ALL_ROWS = ["A3", "A0", "A1", "A2", "A2.5a", "A2.5b"]
MLP_PROJ = ("up_proj", "down_proj")

LADDER_ROWS = {
    "A0": dict(use_grad_gram=False, use_woodbury=True,
               proj_names=MLP_PROJ + ("q_proj", "k_proj", "v_proj")),
    "A1": dict(use_grad_gram=False, use_woodbury=True, proj_names=MLP_PROJ),
    "A2": dict(use_grad_gram=False, use_woodbury=False, proj_names=MLP_PROJ),
    "A3": dict(use_grad_gram=True, use_woodbury=False, proj_names=MLP_PROJ),
}

OFFLADDER_ROWS = {
    "A2.5a": dict(short_side=True),
    "A2.5b": dict(short_side=False),
}


# -----------------------------------------------------------------------------
# Input parser
# -----------------------------------------------------------------------------

def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the K-FAC approximation-ladder influence ablation for one prompt. "
                    "Select the GPU with CUDA_VISIBLE_DEVICES.",
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

    g = p.add_argument_group("Model")
    g.add_argument("--model", type=str, required=True,
                   help=f"Preset name ({', '.join(MODEL_PRESETS)}) or a Hugging Face model id "
                        f"(an HF id uses the generic ModelPreset defaults unless overridden).")
    g.add_argument("--torch_dtype", type=str, default=None,
                   choices=["auto", "float32", "bfloat16", "float16"],
                   help="Model weight dtype (default: from preset).")
    g.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=None,
                   help="Pass trust_remote_code to HF loaders (default: from preset).")

    g = p.add_argument_group("Generation (defaults come from the model preset)")
    g.add_argument("--system_prompt", type=str, default=None,
                   help="System prompt; pass an empty string to disable.")
    g.add_argument("--max_new_tokens", type=int, default=None)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top_k", type=int, default=None, help="Use -1 to leave unset.")
    g.add_argument("--top_p", type=float, default=None)
    g.add_argument("--greedy", action="store_true",
                   help="Greedy decoding (do_sample=False); ignores temperature/top_k/top_p.")
    g.add_argument("--strip_prompt", action=argparse.BooleanOptionalAction, default=None,
                   help="Drop the prompt tokens from the decoded output.")
    g.add_argument("--extract", type=str, default=None,
                   choices=["answer_split", "code_block", "none"],
                   help="Post-processing of the generated text: split on 'Answer:', take the "
                        "first ```python``` block, or keep the raw text.")
    g.add_argument("--seed", type=int, default=0,
                   help="Random seed for generation; use -1 for a non-deterministic run.")

    g = p.add_argument_group("K-FAC / influence")
    g.add_argument("--rows", type=_csv, default=ALL_ROWS,
                   help=f"Comma-separated ablation rows to run, from {ALL_ROWS}.")
    g.add_argument("--baseline_row", type=str, default="A3",
                   help="Row every other row is compared against (always run first).")
    g.add_argument("--rel_eps", type=float, default=0.1,
                   help="Relative epsilon for the KFAC computation.")
    g.add_argument("--min_damping", type=float, default=1e-3,
                   help="KFAC damping floor for the A0-A3 ladder rows.")
    g.add_argument("--offladder_min_damping", type=float, default=1e-5,
                   help="KFAC damping floor for A2.5a/A2.5b; should match the floor hardcoded "
                        "in _precond_params_generic (see ablation plan).")
    g.add_argument("--damping", type=float, default=0.1,
                   help="PreconditioningConfig damping.")
    g.add_argument("--epsilon", type=float, default=1e-3,
                   help="Epsilon parameter for the influence computation.")
    g.add_argument("--if_dtype", type=str, default="float16",
                   help="dtype passed to Influences().")
    g.add_argument("--keep_layers", type=str, default="quarter",
                   help="Layers every row targets: 'quarter' for the model's quarter points "
                        "(also used by A2.5a/A2.5b), or comma-separated indices, e.g. 0,9,18,27. "
                        "Note: A2.5a/A2.5b always use the quarter points.")
    g.add_argument("--fail_fast", action="store_true",
                   help="Raise on the first failing row instead of logging it and continuing.")

    args = p.parse_args(argv)

    # ---- resolve model preset + overrides ----
    preset = MODEL_PRESETS.get(args.model) or ModelPreset(model_id=args.model)
    overrides = {k: getattr(args, k) for k in (
        "torch_dtype", "trust_remote_code", "system_prompt", "max_new_tokens",
        "temperature", "top_k", "top_p", "strip_prompt", "extract",
    ) if getattr(args, k) is not None}
    if overrides.get("top_k") == -1:
        overrides["top_k"] = None
    if overrides.get("system_prompt") == "":
        overrides["system_prompt"] = None
    args.preset = dataclasses.replace(preset, **overrides)

    # ---- validate rows ----
    unknown = [r for r in args.rows + [args.baseline_row] if r not in ALL_ROWS]
    if unknown:
        p.error(f"unknown ablation row(s) {unknown}; choose from {ALL_ROWS}")
    args.rows = [args.baseline_row] + [r for r in args.rows if r != args.baseline_row]

    return args


# -----------------------------------------------------------------------------
# Layer selection
# -----------------------------------------------------------------------------

def quarter_layer_indices(model):
    n_layers = model.config.num_hidden_layers
    return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4})


def resolve_keep_layers(model, spec):
    layers = (quarter_layer_indices(model) if spec == "quarter"
              else sorted({int(x) for x in _csv(spec)}))
    n_layers = model.config.num_hidden_layers
    bad = [i for i in layers if i >= n_layers]
    if bad:
        raise ValueError(f"--keep_layers {bad} out of range for num_hidden_layers={n_layers}")
    return layers


def quarter_layer_pattern(model, gm_for_check=None):
    """Regex name_pattern for A2.5a/A2.5b: MLP up/down proj weights at the
    quarter-point layers of the model, e.g. (0|8|16|24) for 32 layers.
    Cross-checks the indices against gm.get_model_layer_indices() if available.
    """
    quarter_idxs = quarter_layer_indices(model)

    if gm_for_check is not None and hasattr(gm_for_check, "get_model_layer_indices"):
        try:
            available = set(gm_for_check.get_model_layer_indices())
            missing = [i for i in quarter_idxs if i not in available]
            if missing:
                print(
                    f"[warn] quarter-point layer indices {missing} not found among "
                    f"gm.get_model_layer_indices()={sorted(available)}; double-check "
                    f"the layer naming before trusting A2.5a/A2.5b results."
                )
        except Exception as e:
            print(f"[warn] could not verify layer indices via get_model_layer_indices(): {e}")

    idx_pattern = "|".join(str(i) for i in quarter_idxs)
    return rf"model\.layers\.({idx_pattern})\.mlp\.(up_proj|down_proj)\.weight$"


def build_ignore(model, keep_layers, keep_proj):
    """Ignore every Linear except `keep_proj` projections in `keep_layers`."""
    keep_layers, keep_proj = set(keep_layers), tuple(keep_proj)
    ignore = [nn.Embedding, nn.LayerNorm]
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not (m and int(m.group(1)) in keep_layers and name.endswith(keep_proj)):
            ignore.append(mod)
    return ignore


def get_row_setup(row_name, model, args, keep_layers, gm_for_layer_check=None):
    """Returns (gm_kwargs, layers, proj_names) for one ablation row."""
    common = dict(small_matrices=False, normalize=True, use_amp=True, rel_eps=args.rel_eps)

    if row_name in LADDER_ROWS:
        o = LADDER_ROWS[row_name]
        gm_kwargs = dict(
            common,
            use_grad_gram=o["use_grad_gram"],
            use_woodbury=o["use_woodbury"],
            short_side=True,
            name_pattern="default",
            proj_names=o["proj_names"],
            min_damping=args.min_damping,
        )
        return gm_kwargs, keep_layers, o["proj_names"]

    if row_name in OFFLADDER_ROWS:
        o = OFFLADDER_ROWS[row_name]
        gm_kwargs = dict(
            common,
            use_grad_gram=True,       # forced -- generic path always uses gradient Gram
            use_woodbury=False,       # no effect on this code path
            short_side=o["short_side"],
            name_pattern=quarter_layer_pattern(model, gm_for_check=gm_for_layer_check),
            proj_names=MLP_PROJ,
            min_damping=args.offladder_min_damping,
        )
        return gm_kwargs, quarter_layer_indices(model), MLP_PROJ

    raise ValueError(f"Unknown ablation row: {row_name!r}")


# -----------------------------------------------------------------------------
# Comparison against the baseline
# -----------------------------------------------------------------------------

def _norms(t):
    t = t.detach().to(torch.float32)
    return (torch.linalg.matrix_norm(t, ord="fro").item(),
            torch.linalg.matrix_norm(t, ord=2).item())


def compare_blocks(baseline, row, baseline_norms):
    """baseline, row: {block_name: 2-D tensor}. Blocks are matched by name, so
    rows that target different layers/projections are compared only on the
    blocks they share; the rest are listed in "skipped".

    Treating the shared blocks as the diagonal of one block-diagonal matrix,
    returns the exact Frobenius / spectral norm of the difference, plus the
    same norms relative to the baseline (restricted to the shared blocks).
    `baseline_norms` caches per-block baseline (fro, spec) across rows.
    """
    common = [k for k in baseline if k in row]
    skipped = ([f"{k}: baseline only" for k in baseline if k not in row]
               + [f"{k}: row only" for k in row if k not in baseline])

    per_block = {}
    frob_sq = base_frob_sq = 0.0
    spec = base_spec = 0.0
    for k in common:
        a, b = baseline[k], row[k]
        if a.shape != b.shape:
            skipped.append(f"{k}: shape mismatch baseline={tuple(a.shape)} row={tuple(b.shape)}")
            continue
        if k not in baseline_norms:
            baseline_norms[k] = _norms(a)
        bf, bs = baseline_norms[k]
        df, ds = _norms(b - a)
        per_block[k] = {"frobenius_diff": df, "spectral_diff": ds,
                        "frobenius_baseline": bf, "spectral_baseline": bs}
        frob_sq += df ** 2
        base_frob_sq += bf ** 2
        spec = max(spec, ds)
        base_spec = max(base_spec, bs)

    frob, base_frob = frob_sq ** 0.5, base_frob_sq ** 0.5
    return {
        "frobenius_norm": frob,
        "spectral_norm": spec,
        "rel_frobenius_norm": frob / base_frob if base_frob > 0 else float("nan"),
        "rel_spectral_norm": spec / base_spec if base_spec > 0 else float("nan"),
        "n_blocks_compared": len(per_block),
        "skipped": skipped,
        "per_block": per_block,
    }


# -----------------------------------------------------------------------------
# Model loading and generation
# -----------------------------------------------------------------------------

def load_model(preset):
    dtype = preset.torch_dtype if preset.torch_dtype == "auto" else getattr(torch, preset.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        preset.model_id, trust_remote_code=preset.trust_remote_code)
    if preset.disable_default_system_prompt:
        tokenizer.use_default_system_prompt = False

    model = AutoModelForCausalLM.from_pretrained(
        preset.model_id,
        trust_remote_code=preset.trust_remote_code,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def postprocess(text, mode):
    if mode == "answer_split":
        return text.split("Answer:", 1)[1].strip() if "Answer:" in text else text.strip()
    if mode == "code_block":
        matches = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
        if matches:
            return matches[0].strip()
        print("[warn] no ``` code block found in the generation; using the raw text.")
    return text.strip()


def generate_answer(prompt, model, tokenizer, preset, greedy=False):
    messages = []
    if preset.system_prompt:
        messages.append({"role": "system", "content": preset.system_prompt})
    messages.append({"role": "user", "content": prompt})

    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=preset.max_new_tokens, do_sample=not greedy)
    if not greedy:
        for k in ("temperature", "top_k", "top_p"):
            if getattr(preset, k) is not None:
                gen_kwargs[k] = getattr(preset, k)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    if preset.strip_prompt:
        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return postprocess(text, preset.extract)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    preset = args.preset
    os.chdir(args.workdir)

    if args.seed >= 0:
        set_seed(args.seed)

    model, tokenizer = load_model(preset)

    with open(args.train_data, "rb") as f:
        train_data = pickle.load(f)[: args.n_train]

    answer = generate_answer(args.evalPrompt, model, tokenizer, preset, greedy=args.greedy)
    keep_layers = resolve_keep_layers(model, args.keep_layers)
    print(f"Target layers: {keep_layers}")

    base = args.baseline_row
    all_rows_res, errors = {}, {}
    last_ladder_gm = None
    baseline_blocks, baseline_norms = None, {}

    for row_name in args.rows:
        print(f"==== row {row_name} ====")
        gm = if_calc = None
        try:
            if row_name != base and baseline_blocks is None:
                raise RuntimeError(f"baseline row {base} did not complete; nothing to compare against")

            free_cuda()
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            row_start = time.perf_counter()

            gm_kwargs, layers, proj_names = get_row_setup(
                row_name, model, args, keep_layers, gm_for_layer_check=last_ladder_gm)
            config = PreconditioningConfig(
                data_size=1,
                damping=args.damping,
                ema_decay=-1,
                ignore_modules=build_ignore(model, layers, proj_names),
            )
            gm = AdvancedSmallKfacGradientMaker(model, config, **gm_kwargs)

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

            torch.cuda.synchronize(device)
            row_elapsed = time.perf_counter() - row_start
            peak = torch.cuda.max_memory_allocated(device)

            row_blocks = {str(k): v.detach().to("cpu") for k, v in if_calc.conditioned_grads.items()}
            if row_name == base:
                baseline_blocks = row_blocks
            cmp = compare_blocks(baseline_blocks, row_blocks, baseline_norms)

            all_rows_res[row_name] = {
                "wall_clock_sec": row_elapsed,
                "peak_memory_bytes": peak,
                "row_peak_memory_bytes": peak - mem_before,   # excludes model weights etc.
                "target_layers": list(layers),
                "proj_names": list(proj_names),
                f"frobenius_norm_vs_{base}": cmp["frobenius_norm"],
                f"spectral_norm_vs_{base}": cmp["spectral_norm"],
                f"rel_frobenius_norm_vs_{base}": cmp["rel_frobenius_norm"],
                f"rel_spectral_norm_vs_{base}": cmp["rel_spectral_norm"],
                "n_blocks_compared": cmp["n_blocks_compared"],
                "skipped_blocks": cmp["skipped"],
                "per_block": cmp["per_block"],
            }
            print(f"{row_name}: {row_elapsed:.1f}s, row peak {(peak - mem_before) / 2**30:.2f} GiB, "
                  f"{cmp['n_blocks_compared']} blocks compared, "
                  f"rel ||.||_F={cmp['rel_frobenius_norm']:.4g}, "
                  f"rel ||.||_2={cmp['rel_spectral_norm']:.4g}")
            if cmp["skipped"]:
                print(f"[info] {len(cmp['skipped'])} block(s) not compared, e.g. {cmp['skipped'][:3]}")
            if row_name != base and cmp["n_blocks_compared"] == 0:
                print(f"[warn] {row_name} shares no blocks with {base}; check the block names.")

            # Keep the most recent ladder gm so A2.5a/A2.5b can cross-check layer indices.
            if row_name in LADDER_ROWS:
                last_ladder_gm = gm

        except Exception as e:
            if args.fail_fast:
                raise
            print(f"[error] row {row_name} failed:")
            traceback.print_exc()
            errors[row_name] = repr(e)
        finally:
            # Free this row's state so the next row's memory reading is clean.
            del gm, if_calc
            free_cuda()

    # -------------------------------------------------------------------------
    # Save results (merge into existing file, write atomically)
    # -------------------------------------------------------------------------

    filename = args.output_file
    out = {}
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            out = pickle.load(f)

    out[args.evalPrompt] = {
        "baseline_row": base,
        "answer_used": answer,
        "rows": all_rows_res,
        "errors": errors,
        "model_id": preset.model_id,
        "preset": dataclasses.asdict(preset),
        "args": {k: v for k, v in vars(args).items() if k != "preset"},
    }

    tmp = filename + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, filename)
    print(f"Saved results to {os.path.abspath(filename)}")


if __name__ == "__main__":
    main()