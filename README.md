## DAVP – Deep Agentic Variant Prioritization

DAVP is an **LLM‑driven variant prioritization pipeline**.  
Given a rare‑disease case (VCF‑derived table + clinical text / HPO terms), DAVP:

- **Preprocessing**: Input VCF must first be processed with Exomiser and filtered to retain only variants in the top 256 genes.
- **Prelimin8 (Step 1)**: ranks genes using cached gene summaries plus patient‑specific variant snippets.
- **Report writing (Step 2)**: generates rich, per‑variant narrative reports.
- **Elimin8 (Step 3)**: does head‑to‑head LLM comparisons of variant reports to score and rank variants.
- **Tournament (Step 4)**: refines top variants with additional pairwise comparisons for final ranking.

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
- `data/step4_tournament/`: Tournament logs and refined rankings.
- `data/summary/`: per‑sample pipeline summaries (`<SAMPLE>.json`).

Additional resources:

- `gene_cache/`: pre‑computed free‑text gene summaries (one `<GENE>.txt` per gene).
- `variant_reports/`: per‑variant input reports used by Step 2 (if present).
- `benchmarks/`: JSONL benchmark datasets (ClinVar, UDN, etc.) used for evaluation.

The exact input JSONL schema is defined in `davp.py` (e.g. column names like `Gene Name`, `CHROM`, `POS`, `REF`, `ALT`).

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
  - `variant_rank_after_tournament`

Intermediate logs for each step are also written into `data/step*/HG00126.json`.

### Running on all samples

To process all samples in the benchmark:

```bash
python davp.py --all
```

This will run the pipeline on every sample listed in `benchmarks/udn.jsonl` (or your selected benchmark).

---

## 4. Analysis utilities

### 4.1 Ablation analysis

`analyze.py` performs comprehensive ablation analysis by comparing three pipeline variants:

- **DAVP-full**: Complete pipeline (Tournament → Elimin8 → Prelimin8 fallback)
- **DAVP-noTournament**: Skips tournament step (Elimin8 → Prelimin8 fallback)
- **DAVP-prelimin8Only**: Gene-level ranking only (Prelimin8)

Run the analysis:

```bash
python analyze.py
```

The script will:
1. Prompt you to select a benchmark file (e.g., `udn.jsonl`)
2. Aggregate results from all pipeline steps
3. Calculate Top-K recall metrics (Top-1, Top-3, Top-5, Top-10, Top-20)
4. Generate outputs in `analysis/`:
   - `ablation_results.jsonl`: Detailed per-sample rankings
   - `summary_table.csv`: Top-K recall metrics for all methods
   - `gene_ranking_recall.png`: Bar plot of gene ranking performance
   - `variant_ranking_recall.png`: Bar plot of variant ranking performance

The bar plots show grouped bars for each method, making it easy to compare performance across different Top-K thresholds.

---

## 5. Configuration

The default configuration is defined in `davp.py` as `DEFAULT_CONFIG`, with sections:

- `prelimin8`
- `elimin8`
- `tournament`
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
- **Caching**: The pipeline caches step outputs to avoid re-running expensive LLM calls. Delete step output files to force re-computation.

---

Contributions, issues, and suggestions are welcome.
