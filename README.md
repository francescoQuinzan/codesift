# CodeSIFT: Detecting Contaminated Code-Generation Prompt Batches via Influence Functions

Code for **CodeSIFT** (Code Statistical Influence-Function Test), a threat-model-agnostic method for detecting whether a batch of code-generation prompts has been contaminated with prompts that elicit insecure code.

> **Paper:** [*Detecting Contaminated Code-Generation Prompt Batches via Influence Functions*](https://arxiv.org/abs/2608.14303) (arXiv:2608.14303)
> **Data:** [Zenodo, doi:10.5281/zenodo.22662814](https://zenodo.org/records/22662814)

---

## How it works

Rather than looking for known vulnerability patterns, CodeSIFT measures how much a model's parameters would have to move to absorb the code it generates, and tests whether a batch of prompts pushes harder than a benign reference batch does.

1. **Curvature (once per model).** Estimate K-FAC factors for the MLP `up_proj` / `down_proj` weights from a small *curvature dataset* (100 samples from The Stack v2). These are cached and reused for every prompt.
2. **Prompt influence score.** For each prompt `z`, sample `N` completions. For each completion, compute the K-FAC-preconditioned gradient and take its ℓ<sub>p</sub> norm (scaled by ε). Average over the `N` samples to get `IF_{p,ε}(z)`.
3. **Statistical test.** Compare the scores of a candidate batch `D` against a benign reference batch `D*` with a one-sided Welch's *t*-test. If the *p*-value is below α, the batch is flagged as **malicious**.

Because the same approximation is applied to every prompt in `D` and `D*`, systematic approximation bias cancels out in the two-sample comparison.

---

## Repository layout

```text
.
├── src/IF/                     # installable package (import IF)
│   ├── Influence/              # Influences: prompt influence scores
│   ├── KFAC/                   # K-FAC preconditioning
│   │   ├── small/              #   one-sided "small-matrix" K-FAC (used in the paper)
│   │   ├── regular/            #   full two-sided K-FAC
│   │   └── matrix_operations.py
│   ├── Loss/                   # loss functions (incl. Semgrep- and toxicity-based)
│   ├── Data/                   # training-data estimation helpers
│   └── Utils/                  # helpers, incl. the code cleaner used before static analysis
├── experiments/
│   ├── influence.py            # influence norms + NLL / perplexity / entropy for one prompt
│   ├── ablation.py             # K-FAC approximation-ladder ablation
│   ├── detectors.py            # static baselines: Bandit, Semgrep, AST taint tracker
│   └── statistical_test.py     # Welch t-test, AUROC and FPR sweeps (CodeSIFT + baselines)
├── semgrep_instructions/       # off-the-shelf Semgrep rules used by the Semgrep baseline
└── pyproject.toml
```

---

## Installation

Requires Python ≥ 3.11 and a CUDA GPU (see [Hardware](#hardware)).

```bash
git clone <this-repo-url> codesift
cd codesift
pip install -e ".[all]"
```

Install in **editable mode** (`-e`): default paths such as `results/` and `semgrep_instructions/` are resolved relative to the checkout.

<details>
<summary>Optional extras</summary>

| Extra         | Installs                              | Needed for                                 |
|---------------|---------------------------------------|--------------------------------------------|
| `experiments` | `bandit`, `scipy`, `scikit-learn`     | `experiments/detectors.py`, `experiments/statistical_test.py` |
| `quant`       | `bitsandbytes`                        | `experiments/influence.py --load_in_4bit`  |
| `data`        | `datasets`, `sentence-transformers`   | `IF.Data`                                  |

`pip install -e .` installs only the core library. `[all]` installs everything.
</details>

---

## Data

### Prompt datasets

Download `Manuscript_Codes.zip` from [Zenodo](https://zenodo.org/records/22662814) and unpack it:

```bash
mkdir -p data && cd data
curl -L -o Manuscript_Codes.zip \
  "https://zenodo.org/records/22662814/files/Manuscript_Codes.zip?download=1"
unzip Manuscript_Codes.zip
```

The prompts are in `Codes for TDSC/1. Dataset/`:

| File                     | Contents                                  |
|--------------------------|-------------------------------------------|
| `Dataset_Malicious.json` | prompts designed to elicit insecure code  |
| `Dataset_Benign.json`    | ordinary coding prompts (benign reference)|

Each record looks like `{"id", "name", "lang", "content"}`. `content` is the full model input: an instruction plus a Python skeleton with a gap marked `# TODO:`, `# TASK:` or `# INSTRUCTION:`.

The paper's two benchmarks, each with 200 malicious prompts, are:

- **AuthSec:** authentication, authorization and session security (JWT/OAuth misuse, CSRF, IDOR, session and cookie flaws, …)
- **InfraCloud:** infrastructure, cloud and protocol exploitation (SSRF, request smuggling, cache poisoning, CI/CD and IAM abuse, …)

Both are compared against a benign reference pool of 400 prompts drawn from the same domains.

> ⚠️ **Dual use.** The malicious prompts are designed to elicit insecure code. They are released only for research on detecting and preventing insecure code generation. Do not run generated code against real systems.

### Curvature dataset

The K-FAC factors are estimated from 100 samples drawn uniformly at random from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2). `experiments/influence.py` expects them as a pickled **list of strings**, at `datasets/clean_instructions.pkl` by default; change this with `--train_data`. The first `--n_train` entries are used (default 99).

```python
import pickle
samples: list[str] = [...]   # 100 code files from The Stack v2
with open("datasets/clean_instructions.pkl", "wb") as f:
    pickle.dump(samples, f)
```

---

## Reproducing the experiments

The pipeline has three stages: score every prompt, run the static baselines on the same completions, then run the statistical test.

### 1. Influence scores

`experiments/influence.py` scores **one prompt** per call. For that prompt, it:

1. samples `--n_trials` completions;
2. computes the K-FAC-preconditioned gradient norms (ℓ<sub>0.5</sub>, ℓ<sub>1</sub>, ℓ<sub>1.5</sub>, ℓ<sub>2</sub>, ℓ<sub>∞</sub>) of each completion;
3. records the model-intrinsic baselines (NLL, perplexity, mean token entropy) for the same tokens.

Results are merged into `--output_file`, so you can run the script once per prompt over a whole dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/influence.py \
    --model granite \
    --evalPrompt "$(cat prompt.txt)" \
    --n_trials 25 \
    --output_file results/granite-3b-code-instruct-2k_reference_dataset.pkl
```

The output has the form `out[prompt][completion] = {"L1": [...], "L2": [...], ..., "nll": [...], "perplexity": [...], "mean_entropy": [...]}`.

Built-in model presets (`--model`):

| Preset     | Hugging Face model                        | GPUs used in the paper |
|------------|-------------------------------------------|------------------------|
| `granite`  | `ibm-granite/granite-3b-code-instruct-2k` | 1                      |
| `deepseek` | `deepseek-ai/deepseek-coder-6.7b-instruct`| 1                      |
| `qwen`     | `Qwen/Qwen2.5-Coder-7B-Instruct`          | 2 (`--device_map auto`)|

Any other Hugging Face model id also works with generic defaults. Each preset sets its own dtype, sampling settings, system prompt and answer post-processing; every one can be overridden on the command line (`python experiments/influence.py --help`).

### 2. Static baselines

Add Bandit, Semgrep and AST-taint-tracker flags for every completion in a results pickle:

```bash
python experiments/detectors.py --datasetName granite-3b-code-instruct-2k_auth_malicious.pkl
```

`--datasetDir` defaults to `results/`. To check a single snippet without saving anything:

```bash
python experiments/detectors.py --code 'import os; os.system(input())'
```

### 3. Statistical test

```bash
# every model × dataset pair, launched in the background
python experiments/statistical_test.py

# a single pair, in the foreground
python experiments/statistical_test.py --model granite-3b-code-instruct-2k --dataset auth
```

This reads the following files from `--base-dir` (default `results/`):

```text
{model}_reference_dataset.pkl        # benign reference pool D*
{model}_{dataset}_malicious.pkl      # malicious pool, dataset ∈ {auth, infrastructure}
```

It writes `{model}_{dataset}.pkl`, which contains AUROC, FPR, accuracy, F1 and other metrics. These are computed with 1000 bootstrap repetitions and swept over:

- samples per prompt (5–25)
- candidate-set size (10–80)
- injection rate (0.1–1.0)
- significance level α (0.01–0.10)

The model names used in file names are `granite-3b-code-instruct-2k`, `deepseek-coder-6_7b-instruct` and `Qwen2_5-Coder-7B-Instruct`.

### K-FAC ablation (optional)

`experiments/ablation.py` takes the same inputs as `influence.py` and compares levels of the K-FAC approximation.

---

## Hyperparameters

Settings used for the paper's main results (Appendix E):

| Hyperparameter                          | Value                                  | Flag            |
|-----------------------------------------|----------------------------------------|-----------------|
| Upweighting factor ε                    | 1e-3                                   | `--epsilon`     |
| Samples per prompt N                    | 25                                     | `--n_trials`     |
| Base K-FAC damping                      | 0.1                                    | `--damping`     |
| Relative damping coefficient            | 0.1                                    | `--rel_eps`     |
| Damping floor γ: Granite, DeepSeek      | 1e-3                                   | `--min_damping` |
| Damping floor γ: Qwen                   | **1e-2**                               | `--min_damping 1e-2` |
| Projections                             | `up_proj`, `down_proj`                 | `--proj_names`  |
| Curvature samples                       | 100 (The Stack v2)                     | `--n_train`     |
| Significance level α (detection)        | 0.01                                   | —               |
| α sweep (FPR calibration)               | 0.01 – 0.10                            | —               |

---

## Hardware

All experiments ran on one node with 4× NVIDIA RTX A6000 (48 GB) GPUs and CUDA 12.8. Approximate cost per prompt, including K-FAC preconditioning:

| Model                         | Wall-clock (s) | Peak GPU memory (GB) |
|-------------------------------|----------------|----------------------|
| Granite-3B-Code-Instruct-2K   | 32.6           | 30.9                 |
| DeepSeek-Coder-6.7B-Instruct  | 36.8           | 31.6                 |
| Qwen2.5-Coder-7B-Instruct     | 39.0           | 34.6                 |

---

## Citation

```bibtex
@inproceedings{codesift2027,
  title     = {Detecting Contaminated Code-Generation Prompt Batches via Influence Functions},
  author    = {Anonymous},
  booktitle = {Under review at the International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```

## License

Code: see [`LICENSE`](LICENSE). The Zenodo data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
