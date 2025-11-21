from dotenv import load_dotenv
from typing import Dict, Any
from pathlib import Path
import argparse
import logging
import json
import sys

from llm.rankers import RankingInput, RankingItem, Elimin8Comparator, Elimin8Sorter, RoundRobinSorter, RoundRobinComparator

from prompts import PRELIMIN8_PROMPT, FINAL_LLM_PROMPT, ELIMIN8_PROMPT, ROUND_ROBIN_PROMPT
from utils import *

load_dotenv()

DAVP_ROOT = Path(__file__).parent
DAVP_DATA_DIR = DAVP_ROOT / "data"
GENE_CACHE_DIR = DAVP_ROOT / "gene_cache"
REPORTS_DIR = DAVP_ROOT / "variant_reports"

for subdir in ["input", "step1_prelimin8", "step2_reports", "step3_elimin8", "step4_round_robin", "summary"]:
    (DAVP_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "prelimin8": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.0,
        "max_tokens": 8192,
        "max_concurrency": 10,
        "top_k": 10,
        "points": [10, 7, 5, 3, 2, 1, 0, -1],
        "rounds_before_elimination": 1,
        "thinking_budget": 0
    },
    "elimin8": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.0,
        "max_tokens": 8192,
        "max_concurrency": 10,
        "top_k": 8,
        "points": [10, 7, 5, 3, 2, 1, 0, -1],
        "rounds_before_elimination": 1,
        "thinking_budget": 0
    },
    "round_robin": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.0,
        "max_tokens": 8192,
        "max_concurrency": 10,
        "top_k": 4,
        "thinking_budget": 0
    },
    "report_writer": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.0,
        "max_tokens": 8192,
        "max_concurrency": 10,
        "thinking_budget": 0
    }
}




def step1_prelimin8(sample_name: str) -> Dict[str, Any]:
    can_reuse, cached_path = can_reuse_step(sample_name, "step1_prelimin8", DAVP_DATA_DIR)
    if can_reuse:
        logger.info(f"Step 1: Reusing cached output")
        cached_output = load_step_output(sample_name, "step1_prelimin8", DAVP_DATA_DIR, cached_path)
        save_step_output(sample_name, "step1_prelimin8", cached_output, DAVP_DATA_DIR)
        return {"status": "PASSED", "top_genes": cached_output.get("top_genes", []), "answer_gene_rank": cached_output.get("answer_gene_rank"), "session": None}
    
    logger.info("Step 1: Running Prelimin8...")
    sample_df = load_sample_data(sample_name, str(DAVP_DATA_DIR))
    answer_info = get_answer_for_sample(load_answers(str(DAVP_DATA_DIR)), sample_name)
    patient_summary = prepare_patient_summary(answer_info["epicrisis"], answer_info["hpo_inputs"])
    
    gene_dict = create_gene_variant_mapping(sample_df)
    all_genes = list(gene_dict.keys())
    gene_cache = load_gene_cache(str(GENE_CACHE_DIR))
    
    enhanced_gene_reports = {}
    for gene in all_genes:
        if gene in gene_cache:
            gene_variants = sample_df[sample_df["Gene Name"].apply(lambda x: gene in normalize_genes(x))]
            variant_info = f"\n\nPatient Variants in {gene} ({min(5, len(gene_variants))}/{len(gene_variants)}):\n" + "\n".join([create_variant_report(row) for _, row in gene_variants.head(5).iterrows()]) if not gene_variants.empty else ""
            enhanced_gene_reports[gene] = gene_cache[gene] + variant_info
        else:
            enhanced_gene_reports[gene] = f"No report available for {gene}"
    
    p8_cfg = get_step_config("prelimin8", DEFAULT_CONFIG)
    prelimin8_session = create_llm_session("prelimin8", DEFAULT_CONFIG)
    prelimin8_results = Elimin8Sorter.topk(
        RankingInput(items=[RankingItem(id=g, description=r) for g, r in enhanced_gene_reports.items()]),
        k=p8_cfg["top_k"], comparator=Elimin8Comparator(prelimin8_session, PRELIMIN8_PROMPT),
        target=patient_summary, points=p8_cfg["points"], rounds_before_elimination=p8_cfg["rounds_before_elimination"]
    )
    
    top_genes = [item.id for item in prelimin8_results.results]
    gene_present, gene_rank = check_answer_gene_present(answer_info["genes"], top_genes)
    result = {"status": "PASSED" if gene_present else "LOST_AT_PRELIMIN8", "top_genes": top_genes, "answer_gene_rank": gene_rank, "session": prelimin8_session}
    
    if not gene_present:
        logger.warning("Answer gene lost at Prelimin8 stage")
    logger.info(f"Prelimin8: {len(top_genes)} genes passed, answer gene at rank {gene_rank}")
    
    save_step_output(sample_name, "step1_prelimin8", {"sample": sample_name, "step": "step1_prelimin8", "top_genes": top_genes, "answer_gene_rank": gene_rank, "tournament_log": prelimin8_results.model_dump(), "gene_reports": enhanced_gene_reports, "patient_summary": patient_summary}, DAVP_DATA_DIR)
    return result


def step2_reports(sample_name: str) -> Dict[str, Any]:
    can_reuse, cached_path = can_reuse_step(sample_name, "step2_reports", DAVP_DATA_DIR)
    if can_reuse:
        logger.info(f"Step 2: Reusing cached output")
        cached_output = load_step_output(sample_name, "step2_reports", DAVP_DATA_DIR, cached_path)
        save_step_output(sample_name, "step2_reports", cached_output, DAVP_DATA_DIR)
        variant_reports = {}
        if "variant_reports" in cached_output:
            variant_reports = cached_output["variant_reports"]
        elif "responses" in cached_output:
            step2_data = cached_output.get("variants", [])
            variant_ids = [format_variant_id(v["CHROM"], v["POS"], v["REF"], v["ALT"]) for v in step2_data]
            for idx, vid in enumerate(variant_ids):
                if idx < len(cached_output["responses"].get("responses", [])):
                    variant_reports[vid] = cached_output["responses"]["responses"][idx].get("message", {}).get("content", "")
        return {"status": "PASSED", "variant_reports": variant_reports, "variants": cached_output.get("variants", []), "session": None}
    
    logger.info("Step 2: Loading variant reports and generating final reports...")
    sample_df = load_sample_data(sample_name, str(DAVP_DATA_DIR))
    answer_info = get_answer_for_sample(load_answers(str(DAVP_DATA_DIR)), sample_name)
    step1_output = load_step_output(sample_name, "step1_prelimin8", DAVP_DATA_DIR)
    variants_for_ranking = sample_df[sample_df["Gene Name"].apply(lambda x: any(g in normalize_genes(x) for g in step1_output["top_genes"]))].copy()
    variant_present, _ = check_answer_variant_present(answer_info["contig"], answer_info["pos"], answer_info["ref"], answer_info["alt"], variants_for_ranking)
    
    result = {"status": "PASSED" if variant_present else "LOST_AT_REPORTS", "variant_count": len(variants_for_ranking), "answer_variant_present": variant_present}
    if not variant_present:
        logger.warning("Answer variant lost at reports stage")
        return result
    
    logger.info(f"Found {len(variants_for_ranking)} variants in top genes")
    gagi_variant_reports = {}
    for _, variant in variants_for_ranking.iterrows():
        vid = format_variant_id(variant["CHROM"], variant["POS"], variant["REF"], variant["ALT"])
        report_path = REPORTS_DIR / f"{vid}.txt"
        if report_path.exists():
            gagi_variant_reports[vid] = report_path.read_text()
        else:
            logger.warning(f"Report not found for variant {vid}")
            gagi_variant_reports[vid] = "No report found"
    
    logger.info(f"Loaded {len(gagi_variant_reports)} reports from {REPORTS_DIR}")
    report_writer_session = create_llm_session("report_writer", DEFAULT_CONFIG)
    variant_ids_list = [format_variant_id(v["CHROM"], v["POS"], v["REF"], v["ALT"]) for v in clean_variant_dict(variants_for_ranking.to_dict(orient="records"))]
    prompts = [FINAL_LLM_PROMPT.format(variant_info_section=gagi_variant_reports[vid]) for vid in variant_ids_list]
    responses = report_writer_session.batch_generate(prompts)
    variant_reports = {vid: responses.responses[i].content for i, vid in enumerate(variant_ids_list)}
    
    logger.info(f"Generated {len(variant_reports)} final reports")
    save_step_output(sample_name, "step2_reports", {"sample": sample_name, "step": "step2_reports", "variant_reports": variant_reports, "variants": clean_variant_dict(variants_for_ranking.to_dict(orient="records")), "responses": {"responses": [{"message": {"content": r.content}} for r in responses.responses]}}, DAVP_DATA_DIR)
    return {"status": "PASSED", "variant_reports": variant_reports, "variants": clean_variant_dict(variants_for_ranking.to_dict(orient="records")), "session": report_writer_session}


def step3_elimin8(sample_name: str) -> Dict[str, Any]:
    can_reuse, cached_path = can_reuse_step(sample_name, "step3_elimin8", DAVP_DATA_DIR)
    if can_reuse:
        logger.info(f"Step 3: Reusing cached output")
        cached_output = load_step_output(sample_name, "step3_elimin8", DAVP_DATA_DIR, cached_path)
        save_step_output(sample_name, "step3_elimin8", cached_output, DAVP_DATA_DIR)
        return {k: cached_output.get(k, None if "session" in k else []) for k in ["status", "top_variants", "answer_variant_rank", "session"]}
    
    logger.info("Step 3: Running elimin8 ranking...")
    answer_info = get_answer_for_sample(load_answers(str(DAVP_DATA_DIR)), sample_name)
    patient_summary = prepare_patient_summary(answer_info["epicrisis"], answer_info["hpo_inputs"])
    answer_variant = format_variant_id(answer_info["contig"], answer_info["pos"], answer_info["ref"], answer_info["alt"])
    
    step2_output = load_step_output(sample_name, "step2_reports", DAVP_DATA_DIR)
    
    variant_reports = step2_output["variant_reports"].copy()
    gene_cache = load_gene_cache(str(GENE_CACHE_DIR))
    for variant in step2_output["variants"]:
        vid = format_variant_id(variant["CHROM"], variant["POS"], variant["REF"], variant["ALT"])
        if vid in variant_reports:
            gene_names = normalize_genes(variant.get("Gene Name", ""))
            if gene_names:
                variant_reports[vid] += "\n\n--- Gene Reports ---\n" + "\n\n".join([f"=== {g} ===\n{gene_cache.get(g, f'No information available for gene {g}')}" for g in gene_names])
    
    tournament_input = RankingInput(items=[RankingItem(id=vid, description=r) for vid, r in variant_reports.items()])
    elimin8_cfg = get_step_config("elimin8", DEFAULT_CONFIG)
    elimin8_session = create_llm_session("elimin8", DEFAULT_CONFIG)
    elimin8_results = Elimin8Sorter.topk(tournament_input, k=elimin8_cfg["top_k"], comparator=Elimin8Comparator(elimin8_session, ELIMIN8_PROMPT), target=patient_summary, points=elimin8_cfg["points"], rounds_before_elimination=elimin8_cfg["rounds_before_elimination"])
    
    elimin8_top_variants = [item.id for item in elimin8_results.results]
    elimin8_variant_rank = elimin8_top_variants.index(answer_variant) + 1 if answer_variant in elimin8_top_variants else None
    
    result = {"status": "PASSED" if elimin8_variant_rank else "LOST_AT_ELIMIN8", "top_variants": elimin8_top_variants, "answer_variant_rank": elimin8_variant_rank, "session": elimin8_session}
    logger.info(f"Elimin8: {len(elimin8_top_variants)} variants in top {elimin8_cfg['top_k']}, answer variant at rank {elimin8_variant_rank}")
    
    save_step_output(sample_name, "step3_elimin8", {"sample": sample_name, "step": "step3_elimin8", "top_variants": elimin8_top_variants, "answer_variant_rank": elimin8_variant_rank, "elimin8_log": elimin8_results.model_dump(), "variant_reports": {vid: r[:500] + "..." if len(r) > 500 else r for vid, r in variant_reports.items()}}, DAVP_DATA_DIR)
    return result


def step4_round_robin(sample_name: str) -> Dict[str, Any]:
    can_reuse, cached_path = can_reuse_step(sample_name, "step4_round_robin", DAVP_DATA_DIR)
    if can_reuse:
        logger.info(f"Step 4: Reusing cached output")
        cached_output = load_step_output(sample_name, "step4_round_robin", DAVP_DATA_DIR, cached_path)
        save_step_output(sample_name, "step4_round_robin", cached_output, DAVP_DATA_DIR)
        return {k: cached_output.get(k, None if "session" in k else []) for k in ["status", "top_variants", "answer_variant_rank", "session"]}
    
    logger.info("Step 4: Running round_robin ranking...")
    answer_info = get_answer_for_sample(load_answers(str(DAVP_DATA_DIR)), sample_name)
    patient_summary = prepare_patient_summary(answer_info["epicrisis"], answer_info["hpo_inputs"])
    answer_variant = format_variant_id(answer_info["contig"], answer_info["pos"], answer_info["ref"], answer_info["alt"])
    
    step2_output = load_step_output(sample_name, "step2_reports", DAVP_DATA_DIR)
    step3_output = load_step_output(sample_name, "step3_elimin8", DAVP_DATA_DIR)
    
    variant_reports = step2_output["variant_reports"].copy()
    gene_cache = load_gene_cache(str(GENE_CACHE_DIR))
    for variant in step2_output["variants"]:
        vid = format_variant_id(variant["CHROM"], variant["POS"], variant["REF"], variant["ALT"])
        if vid in variant_reports:
            gene_names = normalize_genes(variant.get("Gene Name", ""))
            if gene_names:
                variant_reports[vid] += "\n\n--- Gene Reports ---\n" + "\n\n".join([f"=== {g} ===\n{gene_cache.get(g, f'No information available for gene {g}')}" for g in gene_names])
    
    rr_cfg = get_step_config("round_robin", DEFAULT_CONFIG)
    round_robin_session = create_llm_session("round_robin", DEFAULT_CONFIG)
    round_robin_results = RoundRobinSorter.rank(RankingInput(items=[RankingItem(id=item, description=variant_reports[item]) for item in step3_output["top_variants"]]), comparator=RoundRobinComparator(round_robin_session, ROUND_ROBIN_PROMPT), target=patient_summary)
    
    round_robin_top_variants = [item.id for item in round_robin_results.results[:rr_cfg["top_k"]]]
    round_robin_variant_rank = round_robin_top_variants.index(answer_variant) + 1 if answer_variant in round_robin_top_variants else None
    
    result = {"status": "PASSED" if round_robin_variant_rank else "LOST_AT_ROUND_ROBIN", "top_variants": round_robin_top_variants, "answer_variant_rank": round_robin_variant_rank, "session": round_robin_session}
    logger.info(f"Round Robin: {len(round_robin_top_variants)} variants in top {rr_cfg['top_k']}, answer variant at rank {round_robin_variant_rank}")
    
    save_step_output(sample_name, "step4_round_robin", {"sample": sample_name, "step": "step4_round_robin", "top_variants": round_robin_top_variants, "answer_variant_rank": round_robin_variant_rank, "round_robin_log": round_robin_results.model_dump()}, DAVP_DATA_DIR)
    return result


def run_pipeline(sample_name: str) -> Dict[str, Any]:
    logger.info(f"Starting pipeline for sample: {sample_name}")
    
    result = {
        "sample": sample_name,
        "status": "PASSED",
        "answer_gene": None,
        "answer_variant": None,
        "gene_rank_after_prelimin8": None,
        "variant_rank_after_tournament": None,
        "variant_rank_after_elimin8": None,
        "variant_rank_after_round_robin": None
    }
    
    answers_df = load_answers(str(DAVP_DATA_DIR))
    answer_info = get_answer_for_sample(answers_df, sample_name)
    result["answer_gene"] = answer_info["genes"][0] if answer_info["genes"] else None
    result["answer_variant"] = format_variant_id(answer_info["contig"], answer_info["pos"], answer_info["ref"], answer_info["alt"])
    
    sessions = {}
    
    try:
        step1_result = step1_prelimin8(sample_name)
        result["gene_rank_after_prelimin8"] = step1_result["answer_gene_rank"]
        sessions["prelimin8"] = step1_result.get("session")
        
        if step1_result["status"] != "PASSED":
            result["status"] = step1_result["status"]
            raise Exception(f"Step 1 failed: {step1_result['status']}")
        
        step2_result = step2_reports(sample_name)
        sessions["report_writer"] = step2_result.get("session")
        
        if step2_result["status"] != "PASSED":
            result["status"] = step2_result["status"]
            raise Exception(f"Step 2 failed: {step2_result['status']}")
        
        step3_result = step3_elimin8(sample_name)
        result["variant_rank_after_elimin8"] = step3_result.get("answer_variant_rank")
        sessions["elimin8"] = step3_result.get("session")
        
        if step3_result["status"] != "PASSED":
            result["status"] = step3_result["status"]
            raise Exception(f"Step 3 failed: {step3_result['status']}")
        
        step4_result = step4_round_robin(sample_name)
        result["variant_rank_after_tournament"] = step4_result["answer_variant_rank"]
        result["variant_rank_after_round_robin"] = step4_result.get("answer_variant_rank")
        sessions["round_robin"] = step4_result.get("session")
        
        if step4_result["status"] != "PASSED":
            result["status"] = step4_result["status"]
            raise Exception(f"Step 4 failed: {step4_result['status']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        result["status"] = "FAILED"
        result["error"] = str(e)
    
    finally:
        summary_path = DAVP_DATA_DIR / "summary" / f"{sample_name}.json"

        cleaned_result = {}
        for key, value in result.items():
            if isinstance(value, (list, tuple)):
                cleaned_result[key] = value
            elif isinstance(value, dict):
                cleaned_result[key] = {k: convert_scalar_to_native_type(v) if not isinstance(v, (list, tuple)) else v for k, v in value.items()}
            else:
                cleaned_result[key] = convert_scalar_to_native_type(value)
        
        with open(summary_path, "w") as f:
            json.dump(cleaned_result, f, indent=4)
    
    logger.info(f"Pipeline completed. Status: {result['status']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run DAVP pipeline for a sample")
    parser.add_argument("--sample", required=True, help="Sample name (e.g., HG00118)")
    
    args = parser.parse_args()
    
    try:
        result = run_pipeline(args.sample)
        
        print("\n" + "="*60)
        print(f"DAVP Pipeline Results for {args.sample}")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Answer Gene: {result['answer_gene']}")
        print(f"Answer Variant: {result['answer_variant']}")
        print(f"\nRanks:")
        print(f"  Gene rank after Step 1 (Prelimin8): {result['gene_rank_after_prelimin8']}")
        if result.get('variant_rank_after_elimin8') is not None:
            print(f"  Variant rank after Step 3 (Elimin8): {result['variant_rank_after_elimin8']}")
        if result.get('variant_rank_after_round_robin') is not None:
            print(f"  Variant rank after Step 4 (Round Robin): {result['variant_rank_after_round_robin']}")
        print("="*60)
        
        return 0 if result["status"] == "PASSED" else 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

