# Split_PUF_Artifact

Reproducibility artifact for:

> **A Probabilistic Phase Transition in ML-Based Modeling Attacks on XOR Arbiter PUFs**  

The development repository (with full experiment history) lives at
[inco308/Split_PUF](https://github.com/inco308/Split_PUF). This repository is a
curated subset containing everything needed to reproduce the paper's tables
and figures.

## Directory layout

```
code/
  utils/          PUF simulator and attack models (PUFs.py, data.py, attack.py, ...)
  campaigns/      scripts that generated each result CSV
  analysis/       CI computation, N50 fitting, figure generation
data/             frozen copies of all result CSVs used by the paper
figures/          PDF figures as included in the paper
paper/            LaTeX sources (TCHES)
```

## Environment

All experiments ran on a single NVIDIA GPU (RTX 3080 Ti 12GB, or RTX 4090 24GB
for the large campaigns), Python 3.10, PyTorch 2.1.2, CUDA 12.1. See
`requirements.txt` for the full dependency list.

## Reproducing the tables

Each row of Table 1 (success probabilities) is produced by one campaign
script; run it and append the resulting CSV to `data/`, then recompute the
reported statistics with `code/analysis/analyze.py`. Because fresh PUF
instances are drawn for every run, individual numbers will differ between
runs; success probabilities should fall within the reported 95% confidence
intervals.

| Paper table / figure | Script |
|---|---|
| Table 1, 2-XOR rows | `code/campaigns/extra_campaigns.py` (2xor_baseline.csv) |
| Table 1, 4-XOR rows | `code/campaigns/fix_reviewer_gaps.py` (4xor_N50.csv), `next_experiments.py` |
| Table 1, 6-XOR rows | `baseline_compare.py` (300k–500k), `extra_campaigns.py` (200k), `phase7_variance.py`, `phase8_bootstrap.py` |
| Table 1, 7-XOR rows | `continue_experiments.py` (7xor_baseline.csv), `run_7xor_n50.py` (7xor_N50_fine.csv), `supplement_t2.py` (1.25M runs 6–10) |
| Table 1, 8-XOR rows | `phase7_variance.py`, `phase8_bootstrap.py`, `fix_reviewer_gaps.py` (6M), `extra_campaigns.py` (8M), `replicate_8xor_10M.py` (10M) |
| Table 1, 9-XOR rows | `extra_campaigns.py` (5M, 7.5M, 15M), `auto_experiments_4090.py` (10M, 12M) |
| Table 2 (MLP vs LR) | `baseline_compare.py` |
| Table 3 (architectures) | `attack_8xor.py`, `attack_8xor_v2.py` |
| Table 4 (iPUF) | `run_phase4.py` |
| Evolution baselines | `evolution_baselines.py` (soft-bit NLL objective) |
| Noise grid (Sec. 5.2) | `extra_campaigns.py` (noise_6xor_400k_v2.csv), `supplement_t2.py` (noise_8xor_5M.csv) |
| Polynomial LR | `extra_campaigns.py` (poly_lr_4xor.csv) |
| Figures 1–3 | `code/analysis/make_figures.py` |

## Running a campaign

```bash
pip install -r requirements.txt
cd code/campaigns
python baseline_compare.py          # writes results to ../data/  (see script for CSV path)
```

A few campaign scripts write to a `results/` path relative to the working
directory; run them from the repository root or adjust the CSV path constant
at the top of the script. The frozen CSVs in `data/` are the exact files the
paper's numbers were computed from.

## Computing reported statistics

```bash
cd code/analysis
python analyze.py      # Table 1 CI recomputation + N50 logistic fits
python make_figures.py # regenerates figures/*.pdf
```

## Building the paper

```bash
cd paper
bash build.sh          # needs TeX Live 2026 (iacrj class)
```

`main.tex` uses `iacrj_local.cls` locally (standard `alpha` bibliography
style). For the official TCHES submission, switch back to `iacrj` in the
document class line; the submission system provides `alphaurl.bst`.

## License

Code: MIT. Data and paper: see the paper's license statement.
