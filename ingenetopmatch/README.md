# inGeneTopMatch — Detailed Variant Reports

`inGeneTopMatch` is the knowledge-graph stage of DAVP (paper § Materials & Methods,
*inGeneTopMatch*). For each variant surviving Prelimin8 it runs two halves:

1. **Evidence assembly (graph traversal).** Locate the variant on the genome, collect the
   genomic entities overlapping it (genes, transcripts, regulatory and epigenetic elements),
   and aggregate the ClinVar and GWAS evidence of *other* variants mapped to those entities,
   producing the per-variant **evidence document** `d_i`.
2. **Report generation (one LLM call).** Summarise `d_i` into the narrative **Detailed Variant
   Report** `R_i` — paper Table S1: input "Variant annotations" → output "Variant report",
   one call per surviving variant (Gemini 2.5 Flash, temperature 0.7, thinking disabled).

`R_i` is the document Elimin8 and the Final Tournament then reason over.

This directory makes both halves **runnable outside production**.

## The limitation: GenomicKB cannot be redistributed

In production, `inGeneTopMatch` traverses **GenomicKB**, a knowledge graph of roughly
3.5 × 10⁸ nodes and 1.4 × 10⁹ edges (genes, transcripts, regulatory elements, ontology terms,
GWAS entries, ClinVar records, and 200 bp `chr_chain` segments tiling the genome), served from
a licensed Neo4j deployment. We cannot ship that database. The main demo repository therefore
ships the **precomputed reports** under [`../variant_reports/`](../variant_reports) and
`davp.py` Step 2 loads them, rather than building them from a graph.

## What this directory adds

To show the stage actually working, this package ships **the real algorithm** plus a **small
synthetic graph** for it to run on:

- the inGeneTopMatch traversal / entity-selection / evidence-aggregation / report-assembly
  code, ported from the production `gagi_service` (only the storage backend is swapped —
  Neo4j → a local JSON file);
- [`mini_graph/`](mini_graph): a small, GenomicKB-shaped graph — 2,834 nodes / 2,268 edges
  plus 1,216 ClinVar records (≈600 KB) covering 97 demo variants — seeded from the shipped
  reports so the genes, GWAS traits and ClinVar phenotypes for the demo variants are the real
  ones (full breakdown in [`mini_graph/README.md`](mini_graph/README.md)); and
- the report-generation LLM call (`build_reports --summarize`) that turns `d_i` into `R_i`,
  using the demo's `FINAL_LLM_PROMPT` and Gemini wrapper.

The evidence half is deterministic and runs with no API key; the LLM half is optional and
gated on `GEMINI_API_KEY`.

### Faithful vs. synthetic — what to trust

| | Status |
|---|---|
| Graph traversal, entity selection (overlap + ±10 bp), ClinVar/GWAS interval overlap, report assembly, report format | **Faithful** — ported from production logic |
| Graph **schema** (node/relationship types queried) | **Faithful** — mirrors GenomicKB |
| Graph **scale** (handful of records per entity vs. tens of thousands) | Synthetic / miniature |
| Node **ids** | Synthetic (graph-local, not production Neo4j ids) |
| Gene symbols, GWAS traits, ClinVar phenotype names per demo variant | **Real** (seeded from production reports) |
| Storage backend (JSON file vs. Neo4j) | Different by necessity |

## Module map (→ production `gagi_service`)

| This package | Production file | Role |
|---|---|---|
| `graph_client.py` (`MiniGraph`) | `src/neo4j_base/neo4j_utils.py` + `queries.py` | graph queries (chr_chain by position, neighbours, entities by id, GWAS by rel-id) |
| `entity_selector.py` | `src/services/algorithmic_entity_selector.py` | select entities overlapping the variant |
| `clinvar_analysis.py` | `src/data_retrieval/clinvar_analysis.py` | ClinVar interval overlap |
| `gwas_analysis.py` | `src/data_retrieval/gwass_analysis.py` | GWAS interval overlap + association resolution |
| `report.py` | `src/utils/llm_utils.py` (`prepare_final_llm_prompt`, `format_clinvar_info_for_prompt`) + workflow driver | assemble `d_i`; per-variant pipeline |
| `models.py` | `src/models/models.py` | node / evidence models |
| `prompts.py` | report-format strings | the `variant_reports/*.txt` layout |
| `build_reports.py` (`--summarize`) | `pipeline_steps/gagi_reporter.py` / `final_llm_response` | the `LLM(d_i)` report-generation call |

`process_variant()` in `report.py` runs the same steps as the production single-variant
workflow: locate `chr_chain` tile → fetch neighbours → select overlapping entities →
aggregate GWAS → aggregate ClinVar → assemble `d_i`. `build_reports.py --summarize` then issues
the one report-generation LLM call per variant that turns `d_i` into `R_i`.

## Completeness vs. the paper

The paper defines the stage output as `R_i = ⟨v_i, x_i, c(g_i), d_i, LLM(d_i, P)⟩` with
`d_i = (E_i, {Ont(e), CV(e), GWAS(e), h_e}_{e∈E_i})`. Mapping each term to this implementation:

| Paper term | In this implementation |
|---|---|
| `E_i` — entities with `pos(v_i) ∈ span(e)` | `entity_selector.py` (overlap; production/this port also admit a ±10 bp window) |
| `CV(e)` — ClinVar evidence | `clinvar_analysis.py` |
| `GWAS(e)` — GWAS evidence | `gwas_analysis.py` |
| `Ont(e)` — ontology annotations | surfaced as the EFO/HP/GO/MONDO terms reached via `GWAS(e)`; the production reports do not carry a separate ontology block |
| `h_e` — per-entity feature/epigenetic context | surfaced as the entity **types** in the entity mapping (TF Binding Site, Histone Modification Site, ChromHMM State, …); per-entity signal/tissue attributes are not inlined (production reports omit them too) |
| `x_i` — variant annotation vector | the variant-annotation block of `d_i` |
| `d_i` — evidence document | `report.build_variant_report` |
| `LLM(d_i, P)` — report-generation call | `build_reports.py --summarize` (one call per variant) |
| `c(g_i)` — gene-cache entry | added **downstream** by `davp.py` (Elimin8/Tournament append `gene_cache/`), not inside `d_i` — matching the demo pipeline |

Two faithful-to-code nuances worth flagging: `Ont(e)` and `h_e` are not separate sections in
the production reports (they are folded into the GWAS-linked ontology terms and the entity-type
labels respectively), and the report-writer prompt is **patient-agnostic** — the demo's
`FINAL_LLM_PROMPT` summarises the variant evidence without the HPO set `P`, matching
`gagi_service`. The surrounding ranking stages (Prelimin8, Elimin8, Final Tournament) are
demonstrated by `davp.py` itself.

## Quickstart

```bash
# 1. (optional) regenerate the synthetic graph from the shipped reports
python -m ingenetopmatch.build_mini_graph

# 2. build the evidence documents d_i from the graph for a sample (or --all)
python -m ingenetopmatch.build_reports --sample HG00126
#    -> writes ingenetopmatch/output/HG00126/<variant>.txt

# 2b. also issue the report-generation LLM call (d_i -> narrative R_i); needs GEMINI_API_KEY
python -m ingenetopmatch.build_reports --sample HG00126 --summarize
#    -> additionally writes ingenetopmatch/output/HG00126/<variant>.report.md

# 3. self-check the port against the shipped production reports
python -m ingenetopmatch.verify
```

`build_reports` runs `inGeneTopMatch` on the Prelimin8-surviving variants of a demo sample
(the same set the stage receives in the full pipeline). The evidence half (steps 2 / 3)
requires only `pandas` — no API key, no network. The `--summarize` half (step 2b) additionally
needs the demo's `google-generativeai` dependency and `GEMINI_API_KEY`.

## Correctness

`python -m ingenetopmatch.verify` regenerates every demo variant's report from the synthetic
graph and checks, as hard criteria, that the section structure is correct, the right variant
is rendered, and the graph traversal + ClinVar aggregation ran. On the 97 demo variants it
reports:

```
variants checked                 : 97
PASS/FAIL criteria failures      : 0
[info] variant-annotation block (input-derived, NOT graph-derived) byte-identical to shipped report: 72/97
[info] GWAS / entity-mapping / ClinVar sections are graph-derived and NOT byte-identical (mini-graph is synthetic)
RESULT: PASS
```

**What "byte-identical" does and does not mean.** A report has two kinds of content. The
**variant-annotation block** (rsID, coordinates, ACMG, ClinVar significance, population
frequencies, sequencing) is rendered **straight from the input VCF record and never touches
the graph** — so the shipped report and the regenerated one are formatting the *same input row
with the same formatter*, and match byte-for-byte (72 of 97; the other 25 differ only because
the distributed demo input drifted from the production input — frequency float precision such
as `1.971e-05` vs `1.9710000000000003e-05`, gene-symbol set, or genotype phasing). This metric
therefore checks the **report formatter**, not the graph.

The **graph-derived sections** (GWAS database, entity mapping, ClinVar summary) are produced by
traversing the synthetic graph and are **not** byte-identical to production — the mini-graph is
not the production GenomicKB. For example, the CAPN3 report's ClinVar total is **18** here vs.
**28,200** in production, and the gene's node id is **1741** (graph-local) vs. **16141705**
(production Neo4j). What *is* faithful in these sections is their **structure** and the
**phenotype/trait names** (the builder seeds them from the shipped reports); the evidence
volume, node ids, and exact per-entity associations are synthetic. In short: the algorithm and
report format are reproduced exactly; the graph is a small synthetic stand-in.

## Limitations

- The graph covers only the demo samples' variants (97 loci); a variant outside that coverage
  is reported and skipped.
- Evidence counts are illustrative (mini scale), not the production tens-of-thousands.
- One entity set is materialized per variant rather than sharing gene nodes graph-wide, and
  node ids are graph-local.
- The `--summarize` LLM call needs the demo's `google-generativeai` dependency and a
  `GEMINI_API_KEY`; without it the package builds only the deterministic evidence documents.
