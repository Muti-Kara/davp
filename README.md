## DAVP – Deep Agentic Variant Prioritization

DAVP is an **LLM‑driven variant prioritization pipeline**.  
Given a rare‑disease case (VCF‑derived table + clinical text / HPO terms), DAVP:

- **Preprocessing**: Input VCF must first be processed with Exomiser and filtered to retain only variants in the top 256 genes.
- **Prelimin8 (Step 1)**: ranks genes using cached gene summaries plus patient‑specific variant snippets.
- **Report writing (Step 2)**: generates rich, per‑variant narrative reports.
- **Elimin8 tournament (Step 3)**: does head‑to‑head LLM comparisons of variant reports to score and rank variants.
- **Round‑robin refinement (Step 4)**: optionally re‑ranks top variants with a denser pairwise comparison schedule.

All intermediate artifacts and a final JSON summary (answer gene / variant ranks, status) are written under `data/`.

---

## 1. Installation

Clone the repo and install dependencies (Python ≥ 3.11 recommended):

```bash
git clone git@github.com:Muti-Kara/davp.git
cd davp
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root with at least:

```bash
GEMINI_API_KEY=your_api_key_here
```

The pipeline currently uses the Google Generative AI (Gemini) Python SDK and reads the key from this variable.

---

## 2. Data layout

DAVP expects a `data/` directory with the following subdirectories (created automatically by `davp.py` if missing):

- `data/input/`: input samples in JSONL, one record per sample (e.g. `HG00126.jsonl`). **Note**: Input VCFs must first be processed with Exomiser and filtered to retain only variants in the top 256 genes before conversion to JSONL.
- `data/step1_prelimin8/`: cached outputs from the Prelimin8 gene‑ranking step.
- `data/step2_reports/`: cached final variant reports and variant tables.
- `data/step3_elimin8/`: Elimin8 tournament logs and top‑k variant lists.
- `data/step4_round_robin/`: Round‑robin logs and refined rankings.
- `data/summary/`: per‑sample pipeline summaries (`<SAMPLE>.json`).

Additional resources:

- `gene_cache/`: pre‑computed free‑text gene summaries (one `<GENE>.txt` per gene).
- `variant_reports/`: per‑variant input reports used by Step 2 (if present).
- `benchmarks/`: JSONL benchmark datasets (ClinVar, UDN, etc.) used for evaluation notebooks / scripts.

The exact input JSONL schema is defined in `utils.py` (e.g. column names like `Gene Name`, `CHROM`, `POS`, `REF`, `ALT`).

---

## 3. Running the pipeline

Run DAVP end‑to‑end for a single sample:

```bash
cd davp
python davp.py --sample HG00126
```

On success you get:

- A console summary of the answer gene / variant and their ranks at each stage.
- `data/summary/HG00126.json` with fields like:
  - `status`
  - `answer_gene`
  - `answer_variant`
  - `gene_rank_after_prelimin8`
  - `variant_rank_after_elimin8`
  - `variant_rank_after_round_robin`

Intermediate logs for each step are also written into `data/step*/HG00126.json`.

You can re‑run the pipeline for the same sample; each step checks whether it can **reuse cached outputs** to avoid re‑calling the LLM.

---

## 4. Analysis utilities

### 4.1 CDF plots of ranks

`plot_cdf.py` aggregates ranks across all summaries in `data/summary/` and writes simple empirical CDFs:

```bash
python plot_cdf.py
```

Outputs (in `plots/`):

- `gene_rank_prelimin8_cdf.png`: CDF of `gene_rank_after_prelimin8`.
- `variant_rank_round_robin_cdf.png`: CDF of `variant_rank_after_round_robin`.

These curves are another way to read **top‑k performance** (e.g. the value at rank = 5 is the fraction of samples with the answer in the top‑5).

## 5. Configuration

The default configuration is defined in `davp.py` as `DEFAULT_CONFIG`, with sections:

- `prelimin8`
- `elimin8`
- `round_robin`
- `report_writer`

Each section includes:

- `model`: Gemini model name.
- `temperature`
- `max_tokens`
- `max_concurrency`
- `top_k` (where applicable)
- `points` and `rounds_before_elimination` (for tournament‑style ranking).

---

## 6. Development notes

- **Python version**: target ≥ 3.11.
- **Formatting / style**: standard `black` / `isort` compatible layout; no opinionated tooling is enforced in this repo yet.
- **LLM calls**:
  - Implemented via `llm/session.py` using the Google Generative AI SDK.
  - Batch calls use `ThreadPoolExecutor` for concurrency.
  - Structured outputs are parsed into Pydantic models where appropriate.

---

Contributions, issues, and suggestions are welcome.
