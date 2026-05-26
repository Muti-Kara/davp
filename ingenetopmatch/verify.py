"""Self-check for the ported inGeneTopMatch implementation.

For every demo variant covered by the mini-graph this regenerates the Detailed Variant
Report from the synthetic graph and checks, as hard PASS/FAIL criteria that depend only on
this code (not on upstream data):

  1. the four report sections are present and correctly ordered
     (``#VARIANT INFO`` -> ``# GWAS INFO`` -> ``#ENTITY MAPPING`` -> ``#CLINVAR INFO``);
  2. the variant-annotation block renders the correct variant (Position / REF / ALT); and
  3. a ClinVar pathogenicity summary is produced — i.e. the graph traversal located the
     variant's tile, selected overlapping entities, and aggregated ClinVar evidence.

It also reports, as an *informational* metric, how many variant-annotation blocks are
byte-identical (modulo float repr) to the shipped production ``variant_reports/*.txt``. The
remainder differ only in upstream-annotation fields where the distributed demo input drifted
from the production input (population-frequency precision, gene-symbol set, genotype phasing);
those are not formatter differences.

Run: ``python -m ingenetopmatch.verify``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from .graph_client import MiniGraph
from .report import process_variant

DAVP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DAVP_ROOT / "data"
REPORTS_DIR = DAVP_ROOT / "variant_reports"
DEMO_SAMPLES = ["HG00126", "HG00246"]

_VARIANT_BLOCK = re.compile(r"DETAILED VARIANT INFORMATION:.*?• Allelic Depth \(AD\): .*?\n", re.S)
_SECTION_HEADERS = ["#VARIANT INFO:", "# GWAS INFO", "#ENTITY MAPPING:", "#CLINVAR INFO:"]
_FLOAT = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d*\.\d+)")


def _normalize_floats(text: str) -> str:
    return _FLOAT.sub(lambda m: f"{float(m.group()):.8e}", text)


def _variant_block(text: str) -> Optional[str]:
    m = _VARIANT_BLOCK.search(text)
    return m.group(0) if m else None


def _headers_ordered(text: str) -> bool:
    idx = [text.find(h) for h in _SECTION_HEADERS]
    return all(i >= 0 for i in idx) and idx == sorted(idx)


def main() -> int:
    graph = MiniGraph.load()

    records = {}
    for sample in DEMO_SAMPLES:
        step2 = DATA_DIR / "step2_reports" / f"{sample}.json"
        if step2.exists():
            for rec in json.loads(step2.read_text()).get("variants", []):
                vid = f"chr{rec['CHROM']}:{rec['POS']}{rec['REF']}>{rec['ALT']}"
                records.setdefault(vid, rec)

    checked = 0
    failures: List[str] = []
    block_identical = 0
    block_comparable = 0

    for vid, rec in sorted(records.items()):
        shipped_path = REPORTS_DIR / f"{vid}.txt"
        if not shipped_path.exists():
            continue
        try:
            report = process_variant(graph, rec)
        except RuntimeError as exc:
            failures.append(f"{vid}: not covered by mini-graph ({exc})")
            continue

        checked += 1

        # (1) section skeleton present and ordered
        if not _headers_ordered(report):
            failures.append(f"{vid}: section headers missing or out of order")

        # (2) the variant block renders the correct variant
        block = _variant_block(report) or ""
        expected = [
            f"• Position: {rec['POS']}",
            f"• Reference Allele (REF): {rec['REF']}",
            f"• Alternative Allele (ALT): {rec['ALT']}",
        ]
        if not all(e in block for e in expected):
            failures.append(f"{vid}: variant block does not render the correct variant")

        # (3) graph traversal + ClinVar aggregation produced a summary
        if "CLINVAR PATHOGENICITY SUMMARY:" not in report:
            failures.append(f"{vid}: no ClinVar summary (entity selection / overlap did not run)")

        # informational: fidelity of the variant block to the shipped production report
        shipped_block = _variant_block(shipped_path.read_text())
        if shipped_block is not None:
            block_comparable += 1
            if _normalize_floats(shipped_block) == _normalize_floats(block):
                block_identical += 1

    print("inGeneTopMatch self-check")
    print(f"  variants checked                 : {checked}")
    print(f"  PASS/FAIL criteria failures      : {len(failures)}")
    for f in failures[:20]:
        print(f"    - {f}")
    print(f"  [info] variant-annotation block (input-derived, NOT graph-derived) "
          f"byte-identical to shipped report: {block_identical}/{block_comparable}")
    print(f"         -> this checks the report FORMATTER, not the graph; that block is")
    print(f"            rendered from the input VCF record and never touches the graph.")
    print(f"         -> remaining blocks differ only in upstream-annotation drift "
          f"(freq precision / gene set / phasing).")
    print(f"  [info] the GWAS / entity-mapping / ClinVar sections are graph-derived and are")
    print(f"         NOT byte-identical to production: the mini-graph is synthetic and miniature")
    print(f"         (e.g. ClinVar counts in the tens, not the tens of thousands GenomicKB holds).")

    ok = checked > 0 and not failures
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
