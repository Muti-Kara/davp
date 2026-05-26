"""Run inGeneTopMatch over the mini-graph and write Detailed Variant Reports.

This is the standalone entry point that exercises the ported algorithm. For each
Prelimin8-surviving variant of a sample (the same set that enters inGeneTopMatch in the full
pipeline) it runs the two halves of the stage:

  1. **Evidence assembly** (graph, deterministic): traverse the synthetic ``mini_graph/`` to
     build the per-variant evidence document ``d_i`` and write it to ``output/<sample>/<vid>.txt``.
  2. **Summarisation** (LLM, optional, ``--summarize``): issue the one report-generation call
     per variant the paper specifies (Table S1: input "Variant annotations" -> output
     "Variant report"), turning ``d_i`` into the narrative Detailed Variant Report ``R_i`` and
     writing it to ``output/<sample>/<vid>.report.md``. Uses the demo's ``FINAL_LLM_PROMPT`` and
     ``llm/session.py`` (Gemini); requires ``GEMINI_API_KEY``.

Run:
    python -m ingenetopmatch.build_reports --sample HG00126
    python -m ingenetopmatch.build_reports --all
    python -m ingenetopmatch.build_reports --sample HG00126 --summarize   # also emit R_i
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from .graph_client import MiniGraph
from .report import run_ingenetopmatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingenetopmatch")

DAVP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DAVP_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
DEMO_SAMPLES = ["HG00126", "HG00246"]

# inGeneTopMatch LLM config, per paper Supplementary Table S5: Gemini 2.5 Flash, temperature
# 0.7, extended-reasoning ("thinking") disabled because the task is summarisation, not ranking.
# (The demo's simplified Session has no thinking channel, so it is disabled by construction.)
SUMMARY_MODEL = "gemini-2.5-flash"
SUMMARY_TEMPERATURE = 0.7
SUMMARY_MAX_TOKENS = 8192
SUMMARY_MAX_WORKERS = 10


def load_survivor_variants(sample: str) -> List[Dict[str, Any]]:
    """Load the Prelimin8-surviving variants for a sample (inGeneTopMatch's input set).

    These are recorded in the Step 2 cache produced by the full pipeline.
    """
    step2 = DATA_DIR / "step2_reports" / f"{sample}.json"
    if not step2.exists():
        raise FileNotFoundError(
            f"No Step 2 cache for {sample} at {step2}; cannot determine the inGeneTopMatch input set."
        )
    return json.loads(step2.read_text()).get("variants", [])


def _summarize_reports(reports: Dict[str, str], out_dir: Path) -> int:
    """inGeneTopMatch's report-generation LLM call: d_i -> narrative Variant Report R_i.

    One call per variant via the demo's report-writer prompt (``FINAL_LLM_PROMPT``) and Gemini
    session. Imports are deferred so the package stays importable (and the deterministic graph
    path stays runnable) without an API key or the demo's LLM dependencies.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "--summarize performs the inGeneTopMatch report-generation LLM call and needs "
            "GEMINI_API_KEY. Set it in the environment or .env, or omit --summarize to build "
            "only the evidence documents (d_i)."
        )

    # The demo's FINAL_LLM_PROMPT and llm/ live at the repo root; ensure it is importable.
    repo_root = str(DAVP_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from prompts import FINAL_LLM_PROMPT
    from llm.session import Session

    session = Session(
        model=SUMMARY_MODEL,
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=SUMMARY_MAX_TOKENS,
        max_workers=SUMMARY_MAX_WORKERS,
    )

    vids = list(reports)
    prompts = [FINAL_LLM_PROMPT.format(variant_info_section=reports[vid]) for vid in vids]
    responses = session.batch_generate(prompts)
    for vid, response in zip(vids, responses.responses):
        (out_dir / f"{vid}.report.md").write_text(response.content)
    return len(vids)


def run_for_sample(graph: MiniGraph, sample: str, summarize: bool = False) -> Dict[str, str]:
    variants = load_survivor_variants(sample)
    reports = run_ingenetopmatch(graph, variants)

    out_dir = OUTPUT_DIR / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid, report in reports.items():
        (out_dir / f"{vid}.txt").write_text(report)

    covered = len(reports)
    total = len(variants)
    logger.info("%s: built %d/%d evidence documents (d_i) -> %s", sample, covered, total, out_dir)
    if covered < total:
        logger.info("%s: %d variants not covered by the mini-graph were skipped.", sample, total - covered)

    if summarize:
        n = _summarize_reports(reports, out_dir)
        logger.info("%s: wrote %d narrative Variant Reports (R_i = LLM(d_i)) -> %s", sample, n, out_dir)

    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run inGeneTopMatch over the mini GenomicKB graph.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", help="Sample name (e.g. HG00126).")
    group.add_argument("--all", action="store_true", help="Run on all demo samples.")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Also issue the inGeneTopMatch report-generation LLM call (d_i -> R_i). "
        "Requires GEMINI_API_KEY.",
    )
    args = parser.parse_args(argv)

    graph = MiniGraph.load()
    samples = DEMO_SAMPLES if args.all else [args.sample]
    for sample in samples:
        run_for_sample(graph, sample, summarize=args.summarize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
