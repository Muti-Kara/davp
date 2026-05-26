"""Build the synthetic mini GenomicKB graph used by inGeneTopMatch in this demo.

This script materializes ``mini_graph/graph.json`` + ``mini_graph/clinvar.csv`` so the ported
inGeneTopMatch algorithm has a real graph to traverse. Production reaches the real GenomicKB
(~3.5e8 nodes, ~1.4e9 edges) over Neo4j, which cannot be redistributed; this stand-in is
*seeded from the shipped ``variant_reports/*.txt``* so that the gene symbols, GWAS traits and
ClinVar phenotypes attached to each demo variant are the real ones produced in production —
only the scale (a handful of records per entity instead of tens of thousands) and the node
ids are synthetic.

For each demo variant the builder lays down, around the variant position:
  * a 200 bp ``chr_chain`` tile (the tile ``find_chr_chain_by_position`` will return), two
    flanking 200 bp tiles, and a coarser ``chr_chain`` "region" node;
  * genomic entity nodes (gene / transcript / regulatory / epigenetic) of the types the real
    report listed, each given a span that contains the variant so the selector picks them up;
  * ``phenotype_or_disease`` nodes + GWAS_association variant nodes for the report's GWAS
    traits, positioned in a nested ladder so wider entities accrue more associations (as in
    production); and
  * a small set of ClinVar records carrying the report's real phenotype strings.

Run: ``python -m ingenetopmatch.build_mini_graph``
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DAVP_ROOT = Path(__file__).resolve().parent.parent  # projects/davp
REPORTS_DIR = DAVP_ROOT / "variant_reports"
DATA_DIR = DAVP_ROOT / "data"
OUT_DIR = Path(__file__).parent / "mini_graph"

DEMO_SAMPLES = ["HG00126", "HG00246"]

# Entity span half-widths (bp). The host gene is the widest entity and the only one wide
# enough to overlap the ClinVar records, so ClinVar counts stay clean and predictable; every
# other entity sits inside it, reproducing production's "wider entity -> more evidence" pattern.
HOST_GENE_HALF_WIDTH = 30_000
ENTITY_HALF_WIDTH = {
    "Gene": 18_000,
    "Transcript": 16_000,
    "Non Coding RNA": 8_000,
    "Replication Timing": 18_000,
    "Enhancer": 1_500,
    "Distal Enhancer": 1_500,
    "Proximal Enhancer": 1_500,
    "Super Enhancer": 1_500,
    "Promoter": 600,
    "Insulator": 400,
    "ChromHMM State": 1_000,
    "Histone Modification Site": 500,
    "DNase Hypersensitivity Site": 100,
    "TF Binding Site": 30,
    "TF Binding Motif": 12,
}
DEFAULT_HALF_WIDTH = 300
MAX_ENTITIES_PER_TYPE = 3

# Nested GWAS-variant offsets (bp from the variant): small offsets fall inside narrow
# entities (TF sites, ChromHMM), large ones only inside the wide gene/transcript spans.
GWAS_OFFSET_LADDER = [
    3, -8, 14, -22, 35, -60, 110, -250, 600, -1300,
    3_000, -7_000, 15_000, -28_000, 21_000, -24_000, 9_000, -4_500, 1_800, -900,
]
# ClinVar offsets all sit in (18_000, 30_000): inside the host gene only.
CLINVAR_OFFSETS = [
    20_500, -21_500, 22_500, -23_500, 24_500, -25_500, 26_500, -27_500, 28_500, -29_500,
    20_800, -21_800, 22_800, -23_800, 24_800,
]
MAX_CLINVAR_PHENOTYPES = 15
SIG_CYCLE = ["Pathogenic", "Pathogenic", "Likely pathogenic", "Uncertain significance"]

GWAS_LINE = re.compile(r"^GWAS_([A-Za-z]+_[0-9]+): (.*)$", re.M)
ENTITY_LINE = re.compile(r"^Entity: (.*?) \(ID: (\d+)\)", re.M)
PHENO_LINE = re.compile(r"^• (.*) \(n=(\d+)\)$", re.M)


def _chromosome_sort_key(chr_val):
    s = str(chr_val).upper()
    if s.startswith("CHR"):
        s = s[3:]
    if s.isdigit():
        return (0, int(s))
    return {"X": (1, 23), "Y": (1, 24), "MT": (1, 25)}.get(s, (2, hash(s)))


def parse_report(text: str) -> Dict[str, Any]:
    """Extract the GWAS traits, entity types, and non-benign ClinVar phenotypes from a report."""
    no_gwas = "No GWAS associations found to analyze." in text
    gwas: List[Tuple[str, str, str]] = []
    if not no_gwas:
        for efo_id, rest in GWAS_LINE.findall(text):
            label, _, definition = rest.partition(" - ")
            gwas.append((efo_id, label.strip(), definition.strip()))

    entities = [(etype.strip(), nid) for etype, nid in ENTITY_LINE.findall(text)]

    clinvar_phenos: List[Tuple[str, int]] = []
    for pheno, count in PHENO_LINE.findall(text):
        if pheno.startswith("HP:"):  # HPO-terms block, not a disease/phenotype name
            continue
        clinvar_phenos.append((pheno, int(count)))

    return {"gwas": gwas, "entities": entities, "clinvar_phenos": clinvar_phenos}


def load_demo_variants() -> Dict[str, Dict[str, Any]]:
    """Map variant id -> annotation record for every variant that reaches Step 2 in the demo."""
    variants: Dict[str, Dict[str, Any]] = {}
    for sample in DEMO_SAMPLES:
        step2 = DATA_DIR / "step2_reports" / f"{sample}.json"
        if not step2.exists():
            continue
        data = json.loads(step2.read_text())
        for rec in data.get("variants", []):
            vid = f"chr{rec['CHROM']}:{rec['POS']}{rec['REF']}>{rec['ALT']}"
            variants.setdefault(vid, rec)
    return variants


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = load_demo_variants()

    chr_chains: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    phenotypes: Dict[str, Dict[str, Any]] = {}
    gwas_variants: List[Dict[str, Any]] = []
    clinvar_rows: List[Dict[str, Any]] = []

    node_id = 1_000  # fresh, graph-local id space (production Neo4j ids are not reused)

    def next_id() -> int:
        nonlocal node_id
        node_id += 1
        return node_id

    # One chr_chain node per (chr, start, resolution) window, so nearby variants share tiles
    # instead of creating colliding duplicates.
    tiles_by_key: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    def get_tile(chrom: str, start: int, end: int, resolution: int, gc: float) -> Dict[str, Any]:
        key = (chrom, start, resolution)
        tile = tiles_by_key.get(key)
        if tile is None:
            tile = {
                "id": next_id(), "chr": chrom, "start_loc": start, "end_loc": end,
                "resolution": resolution, "GC_percentage": gc, "neighbor_ids": [],
            }
            tiles_by_key[key] = tile
            chr_chains.append(tile)
        return tile

    n_covered = 0
    for vid in sorted(variants):
        report_path = REPORTS_DIR / f"{vid}.txt"
        if not report_path.exists():
            continue
        variant = variants[vid]
        parsed = parse_report(report_path.read_text())
        chrom = str(variant["CHROM"])
        pos = int(variant["POS"])
        n_covered += 1

        # --- chr_chain tiles: the 200 bp tile spanning the variant + a coarser region node
        # (linked as a neighbour). The region node is what surfaces as a ``chr_chain`` entity.
        tile_start = (pos // 200) * 200
        tile = get_tile(chrom, tile_start, tile_start + 199, 200, 0.41)
        region = get_tile(chrom, pos - 2_500, pos + 2_500, 5_000, 0.41)
        if region["id"] not in tile["neighbor_ids"]:
            tile["neighbor_ids"].append(region["id"])

        # --- genomic entities (gene / transcript / regulatory), seeded from the report types
        gene_symbols = [g for g in (variant.get("Gene Name") or []) if g]
        type_counts: Dict[str, int] = {}
        specs: List[str] = []
        for etype, _ in parsed["entities"]:
            if etype == "chr_chain":  # produced by the region tile, not an explicit entity here
                continue
            if type_counts.get(etype, 0) >= MAX_ENTITIES_PER_TYPE:
                continue
            type_counts[etype] = type_counts.get(etype, 0) + 1
            specs.append(etype)
        if "Gene" not in specs:  # guarantee a wide host gene for ClinVar attachment
            specs.insert(0, "Gene")

        host_assigned = False
        gene_idx = 0
        for etype in specs:
            if etype == "Gene":
                name = gene_symbols[gene_idx] if gene_idx < len(gene_symbols) else (gene_symbols[0] if gene_symbols else "")
                gene_idx += 1
                half = HOST_GENE_HALF_WIDTH if not host_assigned else ENTITY_HALF_WIDTH["Gene"]
                host_assigned = True
            else:
                name = ""
                half = ENTITY_HALF_WIDTH.get(etype, DEFAULT_HALF_WIDTH)
            entities.append({
                "id": next_id(),
                "type": etype,
                "chr": chrom,
                "start_loc": max(1, pos - half),
                "end_loc": pos + half,
                "name": name,
                "props": {"chr_chain_ids": [tile["id"]]},
            })

        # --- GWAS: phenotype_or_disease nodes + association-bearing variant nodes
        for i, (efo_id, label, definition) in enumerate(parsed["gwas"]):
            if efo_id not in phenotypes:
                phenotypes[efo_id] = {"id": efo_id, "label": label, "definition": definition}
            if i < len(GWAS_OFFSET_LADDER):
                off = GWAS_OFFSET_LADDER[i]
            else:
                off = ((i * 7919) % 58_000) - 29_000
            gpos = max(1, pos + (off if off != 0 else 1))
            gwas_variants.append({"id": next_id(), "chr": chrom, "start_loc": gpos, "phenotype_id": efo_id})

        # --- ClinVar: the report's real phenotype strings, at mini scale, inside the host gene
        top_phenos = sorted(parsed["clinvar_phenos"], key=lambda x: x[1], reverse=True)
        top_phenos = top_phenos[:MAX_CLINVAR_PHENOTYPES]
        for j, (pheno, _count) in enumerate(top_phenos):
            off = CLINVAR_OFFSETS[j % len(CLINVAR_OFFSETS)]
            cpos = max(1, pos + off)
            clinvar_rows.append({
                "Chromosome": chrom, "Start": cpos, "Stop": cpos,
                "ClinicalSignificance": SIG_CYCLE[j % len(SIG_CYCLE)],
                "PhenotypeList": pheno, "PhenotypeIDS": "",
            })
        # a few benign records so the pathogenicity summary shows a realistic mix
        for k, boff in enumerate((20_600, -21_100, 22_100)):
            cpos = max(1, pos + boff)
            clinvar_rows.append({
                "Chromosome": chrom, "Start": cpos, "Stop": cpos,
                "ClinicalSignificance": "Benign" if k % 2 == 0 else "Likely benign",
                "PhenotypeList": "not provided", "PhenotypeIDS": "",
            })

    # --- write graph.json (sorted for deterministic output)
    graph = {
        "meta": {
            "description": "Synthetic mini GenomicKB-style graph for the DAVP inGeneTopMatch demo. "
                           "Node ids are graph-local; content (genes, GWAS traits, ClinVar phenotypes) "
                           "is seeded from the shipped variant_reports/*.txt. NOT the production graph.",
            "resolution": 200,
            "n_variants_covered": n_covered,
        },
        "chr_chains": sorted(chr_chains, key=lambda c: c["id"]),
        "entities": sorted(entities, key=lambda e: e["id"]),
        "phenotypes": sorted(phenotypes.values(), key=lambda p: p["id"]),
        "gwas_variants": sorted(gwas_variants, key=lambda v: v["id"]),
    }
    (OUT_DIR / "graph.json").write_text(json.dumps(graph, indent=2))

    # --- write clinvar.csv (sorted by chromosome key then Start, so MiniGraph's bisect works)
    clinvar_rows.sort(key=lambda r: (_chromosome_sort_key(r["Chromosome"]), int(r["Start"])))
    with open(OUT_DIR / "clinvar.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Chromosome", "Start", "Stop", "ClinicalSignificance", "PhenotypeList", "PhenotypeIDS"]
        )
        writer.writeheader()
        writer.writerows(clinvar_rows)

    print(f"Built mini-graph from {n_covered} demo variants:")
    print(f"  chr_chains : {len(graph['chr_chains'])}")
    print(f"  entities   : {len(graph['entities'])}")
    print(f"  phenotypes : {len(graph['phenotypes'])}")
    print(f"  gwas vars  : {len(graph['gwas_variants'])}")
    print(f"  clinvar    : {len(clinvar_rows)}")
    print(f"  -> {OUT_DIR / 'graph.json'}")
    print(f"  -> {OUT_DIR / 'clinvar.csv'}")


if __name__ == "__main__":
    main()
