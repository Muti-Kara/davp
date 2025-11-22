"""Common utility functions for the DAVP pipeline."""

import pandas as pd
import numpy as np
import os
import glob
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from prompts import VARIANT_PROMPT
from llm.session import Session

logger = logging.getLogger(__name__)


def format_hpo_term(hpo_dict: Dict[str, Any]) -> str:
    """Format a single HPO term dictionary into a string."""
    synonyms = ", ".join(hpo_dict.get("synonym", [])) if hpo_dict.get("synonym") else "none"
    return f"{hpo_dict['id']} | {hpo_dict['name']} | {hpo_dict['definition']} | Synonyms: {synonyms}"


def format_hpo_list(hpo_list: List[Dict[str, Any]]) -> str:
    """Format a list of HPO term dictionaries into a newline-separated string."""
    return "\n".join([format_hpo_term(hpo) for hpo in hpo_list])


def format_list(val: Any) -> str:
    """Format a list value into a comma-separated string."""
    if isinstance(val, list) and val:
        return ", ".join(str(v) for v in val)
    return "N/A"


def format_num(val: Any) -> str:
    """Format a numeric value, handling None and NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def create_variant_report(row: pd.Series, rank: Optional[int] = None) -> str:
    """Create a formatted variant report string from a pandas Series row."""
    acmg_str = "N/A"
    acmg = row.get('ACMG')
    if acmg is not None and isinstance(acmg, (list, tuple)) and len(acmg) == 2:
        criteria = ", ".join(acmg[1]) if isinstance(acmg[1], list) else acmg[1]
        acmg_str = f"{acmg[0]} ({criteria})"
    
    spliceai_str = "N/A"
    sai = row.get('SpliceAI')
    if sai and isinstance(sai, list) and len(sai) >= 5:
        spliceai_str = f"{sai[0]} (AG={sai[1]}, AL={sai[2]}, DG={sai[3]}, DL={sai[4]})"
    
    return VARIANT_PROMPT.format(
        rank=rank if rank is not None else "N/A",
        chr=row.get('CHROM', 'N/A'),
        pos=row.get('POS', 'N/A'),
        ref=row.get('REF', 'N/A'),
        alt=row.get('ALT', 'N/A'),
        gene=format_list(row.get('Gene Name')),
        gt=row.get('GT', 'N/A'),
        ids=format_list(row.get('Databases')),
        kg=format_num(row.get('1KG Frequency')),
        gad=format_num(row.get('gnomAD AF')),
        gadg=format_num(row.get('gnomADg AF')),
        acmg=acmg_str,
        clinvar=format_list(row.get('ClinVar SIG')),
        sift=format_num(row.get('SIFT')),
        polyphen=format_num(row.get('Polyphen')),
        cadd=format_num(row.get('CADD')),
        revel=format_num(row.get('REVEL')),
        spliceai=spliceai_str,
        dann=format_num(row.get('DANN')),
        metalr=format_num(row.get('MetalR')),
        am_score=format_num(row.get('Alpha Missense Score')),
        am_pred=row.get('Alpha Missense Prediction') if row.get('Alpha Missense Prediction') else "N/A",
        omim="Yes" if row.get('OMIM') else "N/A"
    )


def freq_filter(sample_df: pd.DataFrame, af_threshold: float = 0.25) -> pd.DataFrame:
    """Filter variants by gnomAD AF threshold."""
    return sample_df[(sample_df["gnomAD AF"] < af_threshold) | sample_df["gnomAD AF"].isna()]


def prepare_patient_summary(epicrisis: str, hpo_inputs: List[Dict[str, Any]]) -> str:
    """Combine epicrisis and HPO terms into a patient summary string."""
    hpo_text = format_hpo_list(hpo_inputs)
    return epicrisis + "\nHPO terms from RAG: \n" + hpo_text


def load_sample_data(sample_name: str, data_dir: str, input_subdir: str = "input") -> pd.DataFrame:
    """Load variant data for a sample from JSONL file."""
    # Try exact match first
    data_path = os.path.join(data_dir, input_subdir, f"{sample_name}.jsonl")
    if os.path.exists(data_path):
        return pd.read_json(data_path, orient="records", lines=True)
    
    # If not found, search for files starting with sample_name
    input_dir = os.path.join(data_dir, input_subdir)
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.startswith(f"{sample_name}_") and filename.endswith(".jsonl"):
                data_path = os.path.join(input_dir, filename)
                return pd.read_json(data_path, orient="records", lines=True)
    
    raise FileNotFoundError(f"Sample data not found for {sample_name} in {data_dir}/{input_subdir}/")


def load_gene_cache(gene_cache_dir: str) -> Dict[str, str]:
    """Load all gene cache files into a dictionary."""
    gene_cache = {}
    for path in glob.glob(os.path.join(gene_cache_dir, "*")):
        gene_name = os.path.basename(path)
        if gene_name.endswith('.txt'):
            gene_name = gene_name[:-4]
        with open(path, "r") as f:
            gene_cache[gene_name] = f.read()
    print(f"Loaded {len(gene_cache)} gene cache files")
    return gene_cache


def load_answers(data_dir: str) -> pd.DataFrame:
    """Load answers from dataset.jsonl file."""
    # Try in input directory first
    dataset_jsonl_path = os.path.join(data_dir, "input", "dataset.jsonl")
    if not os.path.exists(dataset_jsonl_path):
        # Fallback to direct path if provided as full path
        if os.path.exists(data_dir) and data_dir.endswith(".jsonl"):
            dataset_jsonl_path = data_dir
        else:
            raise FileNotFoundError(f"Dataset file not found: {dataset_jsonl_path}")
    return pd.read_json(dataset_jsonl_path, orient="records", lines=True)


def get_answer_for_sample(answers_df: pd.DataFrame, sample_name: str) -> Dict[str, Any]:
    """Get answer information for a specific sample."""
    answer_row = answers_df[answers_df["sample_name"] == sample_name]
    if answer_row.empty:
        raise ValueError(f"No answer found for sample: {sample_name}")
    
    row = answer_row.iloc[0]
    answer_genes = row["genes"].split(",") if isinstance(row["genes"], str) else row["genes"]
    
    return {
        "genes": [g.strip() for g in answer_genes],
        "contig": row["contig"],
        "pos": row["pos"],
        "ref": row["ref"],
        "alt": row["alt"],
        "epicrisis": row["epicrisis"],
        "hpo_inputs": row["hpo_inputs"]
    }


def check_answer_gene_present(answer_genes: List[str], gene_list: List[str]) -> Tuple[bool, Optional[int]]:
    """Check if any answer gene is present in the gene list and return its index (1-based rank)."""
    for idx, gene in enumerate(gene_list, start=1):
        if gene in answer_genes:
            return True, idx
    return False, None


def check_answer_variant_present(
    answer_contig: int, 
    answer_pos: int, 
    answer_ref: str, 
    answer_alt: str,
    variants_df: pd.DataFrame
) -> Tuple[bool, Optional[int]]:
    """Check if answer variant is present in variants dataframe and return its index (1-based rank)."""
    # Normalize dataframe types to strings (and REF/ALT to uppercase) to avoid type mismatches
    vdf = variants_df.copy()
    if 'CHROM' in vdf.columns:
        vdf['CHROM'] = vdf['CHROM'].astype(str)
    if 'POS' in vdf.columns:
        vdf['POS'] = vdf['POS'].astype(str)
    if 'REF' in vdf.columns:
        vdf['REF'] = vdf['REF'].astype(str).str.upper()
    if 'ALT' in vdf.columns:
        vdf['ALT'] = vdf['ALT'].astype(str).str.upper()
    
    ans_contig = str(answer_contig)
    ans_pos = str(answer_pos)
    ans_ref = str(answer_ref).upper()
    ans_alt = str(answer_alt).upper()
    
    for idx, (_, row) in enumerate(vdf.iterrows(), start=1):
        if (
            row.get('CHROM') == ans_contig and
            row.get('POS') == ans_pos and
            row.get('REF') == ans_ref and
            row.get('ALT') == ans_alt
        ):
            return True, idx
    return False, None


def format_variant_id(contig: int, pos: int, ref: str, alt: str) -> str:
    """Format variant information into a standard ID string."""
    return f"chr{contig}:{pos}{ref}>{alt}"


def normalize_genes(gene_names: Any) -> List[str]:
    """Normalize gene names from various formats to a list of strings."""
    if isinstance(gene_names, list):
        return [g.strip() for g in gene_names if isinstance(g, str) and g.strip()]
    if isinstance(gene_names, str) and gene_names:
        return [g.strip() for g in gene_names.split(',') if g.strip()]
    return []


def create_gene_variant_mapping(sample_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create mapping from genes to variant info strings.
    Returns {gene_name: [variant_info_strings]}
    """
    gene_dict: Dict[str, List[str]] = {}
    
    for rank, (idx, row) in enumerate(sample_df.iterrows(), start=1):
        gene_names = normalize_genes(row.get('Gene Name'))
        if not gene_names:
            continue
        
        variant_info = create_variant_report(row, rank)
        
        for gene in gene_names:
            if gene not in gene_dict:
                gene_dict[gene] = []
            gene_dict[gene].append(variant_info)
    
    return gene_dict


def get_step_config(step_name: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    step_config_map = {
        "prelimin8": "prelimin8",
        "elimin8": "elimin8",
        "round_robin": "round_robin",
        "report_writer": "report_writer"
    }
    config_key = step_config_map.get(step_name)
    if config_key:
        return default_config.get(config_key, {})
    return {}


def get_step_file_path(sample_name: str, step_name: str, data_dir: Path) -> Path:
    return data_dir / step_name / f"{sample_name}.json"


def can_reuse_step(sample_name: str, step_name: str, data_dir: Path) -> Tuple[bool, Optional[Path]]:
    if step_name == "reports":
        return False, None  # Reports are always read from in_gene_top_match
    file_path = get_step_file_path(sample_name, step_name, data_dir)
    if file_path.exists():
        logger.info(f"Found cached output for step '{step_name}' at {file_path}")
        return True, file_path
    return False, None


def load_step_output(sample_name: str, step_name: str, data_dir: Path, file_path: Optional[Path] = None) -> Dict[str, Any]:
    if file_path is None:
        file_path = get_step_file_path(sample_name, step_name, data_dir)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Step output not found: {file_path}")
    
    with open(file_path, "r") as f:
        return json.load(f)


def convert_scalar_to_native_type(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif pd.isna(value):
        return None
    else:
        return value


def clean_variant_dict(variants_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for variant in variants_dict:
        for key, value in variant.items():
            if isinstance(value, (list, tuple)):
                continue
            else:
                variant[key] = convert_scalar_to_native_type(value)
    return variants_dict


def save_step_output(sample_name: str, step_name: str, data: Dict[str, Any], data_dir: Path) -> Path:
    file_path = get_step_file_path(sample_name, step_name, data_dir)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            cleaned_data[key] = value
        elif isinstance(value, dict):
            cleaned_data[key] = {k: convert_scalar_to_native_type(v) if not isinstance(v, (list, tuple)) else v for k, v in value.items()}
        else:
            cleaned_data[key] = convert_scalar_to_native_type(value)
    
    with open(file_path, "w") as f:
        json.dump(cleaned_data, f, indent=4) 
    
    return file_path


def create_llm_session(step_name: str, default_config: Dict[str, Any]):
    """Create a Session using default config for the step."""
    cfg = get_step_config(step_name, default_config)
    return Session(
        model=cfg.get("model", "gemini-2.0-flash-exp"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 8192),
        max_workers=cfg.get("max_concurrency", 10)
    )

