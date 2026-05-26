"""Detailed Variant Report assembly + the inGeneTopMatch driver.

``build_variant_report`` assembles the per-variant evidence document ``d_i`` (variant
annotations + GWAS database + entity->GWAS mapping + ClinVar summary) in the exact layout of
the shipped ``variant_reports/*.txt``. ``run_ingenetopmatch`` is the per-variant pipeline:
locate the variant's ``chr_chain`` tile, pull its neighbours, select overlapping entities,
aggregate GWAS + ClinVar evidence, and emit the report. This is a direct port of the
``gagi_service`` ``create_neo4j_genomic_workflow`` graph + ``prepare_final_llm_prompt``
report-assembly path; production runs it as one parallel call per surviving variant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .clinvar_analysis import analyze_by_clinvar
from .entity_selector import select_entities_algorithmically
from .graph_client import MiniGraph
from .gwas_analysis import analyze_by_gwas
from .models import ReportedNeighbor
from .prompts import (
    CLINVAR_FORMAT,
    CLINVAR_HEADER,
    ENTITY_HEADER,
    GWAS_HEADER,
    NO_GWAS_TEXT,
    REPORT_HEADER,
    VARIANT_FORMAT,
)

logger = logging.getLogger(__name__)


def format_variant_id(chrom, pos, ref, alt) -> str:
    return f"chr{chrom}:{pos}{ref}>{alt}"


# --------------------------------------------------------------------- variant block


def _join(val: Any) -> str:
    if isinstance(val, list) and val:
        return ", ".join(str(v) for v in val)
    return "N/A"


def _scalar(val: Any) -> Any:
    return val if val not in (None, "", [], "N/A") else "N/A"


def format_variant_info(variant: Dict[str, Any]) -> str:
    """Render the variant annotation block (x_i) from a raw input record."""
    databases = variant.get("Databases") or []
    rsid = next((db for db in databases if isinstance(db, str) and db.startswith("rs")), None)
    rsid = rsid or variant.get("rsid") or "N/A"

    acmg_val = variant.get("ACMG")
    if isinstance(acmg_val, list) and acmg_val:
        acmg = acmg_val[0]
    elif acmg_val:
        acmg = str(acmg_val)
    else:
        acmg = "N/A"

    omim_val = variant.get("OMIM")
    omim = str(omim_val) if omim_val else "N/A"

    ad_val = variant.get("AD")
    ad = str(ad_val) if ad_val is not None else "N/A"

    return VARIANT_FORMAT.format(
        rsid=rsid,
        chrom=variant.get("CHROM", "N/A"),
        pos=variant.get("POS", "N/A"),
        ref=variant.get("REF", "N/A"),
        alt=variant.get("ALT", "N/A"),
        gt=_scalar(variant.get("GT")),
        qual=_scalar(variant.get("QUAL")),
        gene_names=_join(variant.get("Gene Name")),
        databases=_join(databases),
        clinvar_sig=_join(variant.get("ClinVar SIG")),
        acmg=acmg,
        omim=omim,
        gnomad_af=_scalar(variant.get("gnomAD AF")),
        gnomadg_af=_scalar(variant.get("gnomADg AF")),
        kg_frequency=_scalar(variant.get("1KG Frequency")),
        dp=_scalar(variant.get("DP")),
        ad=ad,
    )


# --------------------------------------------------------------------- GWAS + entity blocks


def format_gwas_and_entities(neighbors: List[ReportedNeighbor]) -> Tuple[str, str]:
    """Build the GWAS database + entity->GWAS mapping blocks.

    Direct port of the GWAS/entity portion of production ``prepare_final_llm_prompt``:
    duplicate GWAS references within an entity are preserved (one per association), and the
    unique-GWAS database lists each phenotype/disease once.
    """
    filtered = [n for n in neighbors if n.gwas_associations]

    unique_gwas: Dict[str, Dict[str, str]] = {}
    entity_gwas_mapping: Dict[str, Dict[str, Any]] = {}

    for neighbor in filtered:
        entity_gwas_ids: List[str] = []
        for trio in neighbor.gwas_associations:
            phenotype = trio.phenotype
            gwas_id = phenotype.id
            if gwas_id not in unique_gwas:
                unique_gwas[gwas_id] = {
                    "id": gwas_id,
                    "label": phenotype.label,
                    "definition": phenotype.definition,
                }
            entity_gwas_ids.append(gwas_id)
        if entity_gwas_ids:
            entity_gwas_mapping[f"{neighbor.type}_{neighbor.id}"] = {
                "type": neighbor.type,
                "id": neighbor.id,
                "gwas_ids": entity_gwas_ids,
            }

    if unique_gwas:
        gwas_info = "GWAS ASSOCIATIONS DATABASE:\n"
        for gwas_id, details in unique_gwas.items():
            gwas_info += f"GWAS_{details['id']}: {details['label']} - {details['definition']}\n"
        gwas_info += "\n"
    else:
        gwas_info = NO_GWAS_TEXT

    entity_mapping = ""
    if entity_gwas_mapping:
        entity_mapping = "GENETIC ENTITIES AND THEIR ASSOCIATED GWAS:\n"
        for entity_info in entity_gwas_mapping.values():
            gwas_refs = ", ".join(f"GWAS_{gid}" for gid in entity_info["gwas_ids"])
            entity_mapping += (
                f"Entity: {entity_info['type']} (ID: {entity_info['id']}) "
                f"-> Associated GWAS: {gwas_refs}\n"
            )
        entity_mapping += "\n"

    return gwas_info, entity_mapping


# --------------------------------------------------------------------- ClinVar block


def format_clinvar_info(neighbors: List[ReportedNeighbor]) -> str:
    """Build the ClinVar pathogenicity summary + non-benign phenotype breakdown.

    Direct port of production ``format_clinvar_info_for_prompt``.
    """
    if not neighbors:
        return ""

    total = pathogenic = benign = uncertain = other = 0
    phenotype_count: Dict[str, int] = {}
    hpo_count: Dict[str, int] = {}

    for neighbor in neighbors:
        cv = getattr(neighbor, "clinvar_associations", None)
        if cv is None or cv.empty:
            continue
        total += len(cv)
        for _, row in cv.iterrows():
            sig = str(row.get("ClinicalSignificance", "")).lower()
            if "pathogenic" in sig and "likely" not in sig:
                pathogenic += 1
            elif "likely pathogenic" in sig:
                pathogenic += 1
            elif "benign" in sig and "likely" not in sig:
                benign += 1
            elif "likely benign" in sig:
                benign += 1
            elif "uncertain" in sig or "conflicting" in sig:
                uncertain += 1
            else:
                other += 1

            is_benign = ("benign" in sig and "likely" not in sig) or ("likely benign" in sig)
            if is_benign:
                continue

            phenotypes = row.get("PhenotypeList", "")
            if phenotypes and str(phenotypes) != "nan" and str(phenotypes).strip():
                for pheno in str(phenotypes).split(";"):
                    pheno = pheno.strip()
                    if pheno and pheno.lower() not in ["not provided", "not specified", ""]:
                        phenotype_count[pheno] = phenotype_count.get(pheno, 0) + 1

            hpo_ids = row.get("PhenotypeIDS", "")
            if hpo_ids and str(hpo_ids) != "nan" and str(hpo_ids).strip():
                for hpo in str(hpo_ids).split(";"):
                    hpo = hpo.strip()
                    if hpo and hpo.startswith("HP:"):
                        hpo_count[hpo] = hpo_count.get(hpo, 0) + 1

    if total == 0:
        return ""

    clinvar_info = CLINVAR_FORMAT.format(
        total_variants=total,
        pathogenic_variants=pathogenic,
        benign_variants=benign,
        uncertain_variants=uncertain,
        other_variants=other,
    )

    if phenotype_count:
        clinvar_info += "ASSOCIATED PHENOTYPES/DISEASES (NON-BENIGN VARIANTS):\n"
        clinvar_info += "====================================================\n"
        for phenotype, count in sorted(phenotype_count.items(), key=lambda x: x[1], reverse=True):
            clinvar_info += f"• {phenotype} (n={count})\n"
        clinvar_info += "\n"

    if hpo_count:
        clinvar_info += "HPO TERMS (NON-BENIGN VARIANTS):\n"
        clinvar_info += "================================\n"
        for hpo_term, count in sorted(hpo_count.items(), key=lambda x: x[1], reverse=True):
            clinvar_info += f"• {hpo_term} (n={count})\n"
        clinvar_info += "\n"

    return clinvar_info


# --------------------------------------------------------------------- assembly + driver


def build_variant_report(variant: Dict[str, Any], neighbors: List[ReportedNeighbor]) -> str:
    """Assemble the full Detailed Variant Report (``d_i``) for one variant."""
    variant_id = format_variant_id(
        variant.get("CHROM"), variant.get("POS"), variant.get("REF"), variant.get("ALT")
    )
    variant_info = format_variant_info(variant)
    gwas_info, entity_mapping = format_gwas_and_entities(neighbors)
    clinvar_info = format_clinvar_info(neighbors)

    return (
        REPORT_HEADER.format(variant_id=variant_id)
        + variant_info
        + GWAS_HEADER
        + gwas_info
        + ENTITY_HEADER
        + entity_mapping
        + CLINVAR_HEADER
        + clinvar_info
    )


def process_variant(graph: MiniGraph, variant: Dict[str, Any]) -> str:
    """Run the full inGeneTopMatch traversal for a single variant and return its report.

    Mirrors the production single-variant workflow: locate chr_chain tile -> fetch
    neighbours -> select overlapping entities -> aggregate GWAS -> aggregate ClinVar ->
    assemble the Detailed Variant Report.
    """
    chrom = variant.get("CHROM")
    pos = int(variant.get("POS"))

    chain = graph.find_chr_chain_by_position(chrom, pos)
    if chain is None:
        raise RuntimeError(
            f"No chr_chain tile covers {chrom}:{pos} in the mini-graph "
            f"(region not in the synthetic graph's coverage)."
        )

    neighbours = graph.get_chr_chain_neighbours(chain.id)
    selected = select_entities_algorithmically(chrom, pos, neighbours)
    analyze_by_gwas(graph, selected)
    analyze_by_clinvar(graph, selected)
    return build_variant_report(variant, selected)


def run_ingenetopmatch(
    graph: MiniGraph,
    variants: List[Dict[str, Any]],
    only_variant_ids: Optional[set] = None,
) -> Dict[str, str]:
    """Build Detailed Variant Reports for a list of variants.

    Production issues these as one parallel report-generation call per surviving variant;
    the traversal is deterministic and side-effect-free, so here it runs sequentially.
    Variants the mini-graph does not cover are logged and skipped rather than aborting.
    """
    reports: Dict[str, str] = {}
    for variant in variants:
        vid = format_variant_id(
            variant.get("CHROM"), variant.get("POS"), variant.get("REF"), variant.get("ALT")
        )
        if only_variant_ids is not None and vid not in only_variant_ids:
            continue
        try:
            reports[vid] = process_variant(graph, variant)
        except RuntimeError as exc:
            logger.warning("inGeneTopMatch skipped %s: %s", vid, exc)
    return reports
