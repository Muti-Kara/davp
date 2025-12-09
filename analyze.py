#!/usr/bin/env python3
"""
DAVP Analysis Script

This script analyzes DAVP pipeline results by:
1. Loading benchmark answers and pipeline outputs
2. Extracting gene/variant rankings from intermediate step files
3. Computing ablation variants (full, no-tournament, prelimin8-only)
4. Generating simple performance metrics and plots

Based on the aggregation pattern from the full GAGI analysis but simplified
for demonstration purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Paths
DAVP_ROOT = Path(__file__).parent
DATA_DIR = DAVP_ROOT / "data"
BENCHMARKS_DIR = DAVP_ROOT / "benchmarks"
OUTPUT_DIR = DAVP_ROOT / "analysis"

# Step output templates (as strings for formatting)
STEP1_TEMPLATE = str(DATA_DIR / "step1_prelimin8" / "{sample_name}.json")
STEP3_TEMPLATE = str(DATA_DIR / "step3_elimin8" / "{sample_name}.json")
STEP4_TEMPLATE = str(DATA_DIR / "step4_tournament" / "{sample_name}.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'


def get_variant_id(contig: Any, pos: Any, ref: str, alt: str) -> str:
    """Format variant as chr:pos:ref>alt"""
    return f"chr{contig}:{pos}{ref}>{alt}"


def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark JSONL file"""
    answers = []
    with open(benchmark_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Normalize to list format
                genes = data["genes"] if isinstance(data["genes"], list) else [data["genes"]]
                answers.append({
                    "sample_name": data["sample_name"],
                    "variant": get_variant_id(data["contig"], data["pos"], data["ref"], data["alt"]),
                    "gene": genes,
                    "type": data.get("type", "unknown")
                })
    return answers


def load_input_variants(sample_name: str) -> pd.DataFrame:
    """Load input variants for a sample"""
    input_path = DATA_DIR / "input" / f"{sample_name}.jsonl"
    if not input_path.exists():
        return pd.DataFrame()
    return pd.read_json(input_path, orient="records", lines=True)


def aggregate_prelimin8_results(sample_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate Prelimin8 gene rankings from step1 outputs.
    Returns: {sample_name: {"top_genes": [...], "answer_gene_rank": int}}
    """
    print("Aggregating Prelimin8 results...")
    results = {}
    
    for sample_name in sample_names:
        step1_path = Path(STEP1_TEMPLATE.format(sample_name=sample_name))
        if not step1_path.exists():
            continue
        
        with open(step1_path, "r") as f:
            step1_data = json.load(f)
        
        results[sample_name] = {
            "top_genes": step1_data.get("top_genes", []),
            "answer_gene_rank": step1_data.get("answer_gene_rank")
        }
    
    return results


def aggregate_elimin8_results(sample_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate Elimin8 variant rankings from step3 outputs.
    Returns: {sample_name: {"top_variants": [...], "answer_variant_rank": int}}
    """
    print("Aggregating Elimin8 results...")
    results = {}
    
    for sample_name in sample_names:
        step3_path = Path(STEP3_TEMPLATE.format(sample_name=sample_name))
        if not step3_path.exists():
            continue
        
        with open(step3_path, "r") as f:
            step3_data = json.load(f)
        
        results[sample_name] = {
            "top_variants": step3_data.get("top_variants", []),
            "answer_variant_rank": step3_data.get("answer_variant_rank")
        }
    
    return results


def aggregate_tournament_results(sample_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate Tournament variant rankings from step4 outputs.
    Returns: {sample_name: {"top_variants": [...], "answer_variant_rank": int}}
    """
    print("Aggregating Tournament results...")
    results = {}
    
    for sample_name in sample_names:
        step4_path = Path(STEP4_TEMPLATE.format(sample_name=sample_name))
        if not step4_path.exists():
            continue
        
        with open(step4_path, "r") as f:
            step4_data = json.load(f)
        
        results[sample_name] = {
            "top_variants": step4_data.get("top_variants", []),
            "answer_variant_rank": step4_data.get("answer_variant_rank")
        }
    
    return results


def convert_to_gene_rankings(variant_results: Dict, input_variants: Dict[str, pd.DataFrame], answers: List[Dict]) -> Dict:
    """Convert variant rankings to gene rankings by extracting genes from variants"""
    gene_results = {}
    
    for sample_name in variant_results:
        if sample_name not in input_variants:
            continue
        
        input_df = input_variants[sample_name]
        
        # Build variant -> genes mapping
        variant_to_genes = {}
        for _, row in input_df.iterrows():
            variant_id = get_variant_id(row["CHROM"], row["POS"], row["REF"], row["ALT"])
            genes = row.get("Gene Name", [])
            if not isinstance(genes, list):
                genes = [genes] if genes else []
            variant_to_genes[variant_id] = genes
        
        # Get answer genes for this sample
        answer_genes = set()
        for answer in answers:
            if answer["sample_name"] == sample_name:
                answer_genes.update(answer["gene"])
        
        # Extract genes in order of variant appearance
        seen_genes = set()
        gene_list = []
        top_variants = variant_results[sample_name].get("top_variants", [])
        
        for variant in top_variants:
            variant_genes = set(variant_to_genes.get(variant, []))
            # Prioritize answer genes
            answer_genes_in_variant = variant_genes.intersection(answer_genes)
            
            for gene in list(answer_genes_in_variant) + list(variant_genes):
                if gene not in seen_genes:
                    seen_genes.add(gene)
                    gene_list.append(gene)
        
        gene_results[sample_name] = {"top_genes": gene_list}
    
    return gene_results


def extract_gene_ranking(sample_name: str, results: Dict, target_gene: str) -> int:
    """Extract rank of target gene from results"""
    if sample_name not in results:
        return None
    
    top_genes = results[sample_name].get("top_genes", [])
    try:
        return top_genes.index(target_gene) + 1
    except ValueError:
        return None


def extract_variant_ranking(sample_name: str, results: Dict, target_variant: str) -> int:
    """Extract rank of target variant from results"""
    if sample_name not in results:
        return None
    
    top_variants = results[sample_name].get("top_variants", [])
    try:
        return top_variants.index(target_variant) + 1
    except ValueError:
        return None


def create_ablation_dataframe(answers: List[Dict], 
                               prelimin8_results: Dict,
                               elimin8_results: Dict,
                               tournament_results: Dict,
                               elimin8_gene_results: Dict,
                               tournament_gene_results: Dict) -> pd.DataFrame:
    """
    Create ablation analysis dataframe with three variants:
    - DAVP-full: tournament -> elimin8 -> prelimin8 fallback
    - DAVP-noTournament: elimin8 -> prelimin8 fallback
    - DAVP-prelimin8Only: prelimin8 only
    """
    print("Creating ablation dataframe...")
    rows = []
    
    for answer in answers:
        sample_name = answer["sample_name"]
        genes = answer["gene"]
        variant = answer["variant"]
        
        # DAVP-full gene rank (tournament -> elimin8 -> prelimin8)
        gene_rank_full = None
        for gene in genes:
            rank = extract_gene_ranking(sample_name, tournament_gene_results, gene)
            if rank is not None:
                gene_rank_full = rank
                break
        if gene_rank_full is None:
            for gene in genes:
                rank = extract_gene_ranking(sample_name, elimin8_gene_results, gene)
                if rank is not None:
                    gene_rank_full = rank
                    break
        if gene_rank_full is None:
            for gene in genes:
                rank = extract_gene_ranking(sample_name, prelimin8_results, gene)
                if rank is not None:
                    gene_rank_full = rank
                    break
        
        # DAVP-full variant rank (tournament -> elimin8)
        variant_rank_full = extract_variant_ranking(sample_name, tournament_results, variant)
        if variant_rank_full is None:
            variant_rank_full = extract_variant_ranking(sample_name, elimin8_results, variant)
        
        # DAVP-noTournament gene rank (elimin8 -> prelimin8)
        gene_rank_noTournament = None
        for gene in genes:
            rank = extract_gene_ranking(sample_name, elimin8_gene_results, gene)
            if rank is not None:
                gene_rank_noTournament = rank
                break
        if gene_rank_noTournament is None:
            for gene in genes:
                rank = extract_gene_ranking(sample_name, prelimin8_results, gene)
                if rank is not None:
                    gene_rank_noTournament = rank
                    break
        
        # DAVP-noTournament variant rank (elimin8 only)
        variant_rank_noTournament = extract_variant_ranking(sample_name, elimin8_results, variant)
        
        # DAVP-prelimin8Only gene rank (prelimin8 only)
        gene_rank_prelimin8Only = None
        for gene in genes:
            rank = extract_gene_ranking(sample_name, prelimin8_results, gene)
            if rank is not None:
                gene_rank_prelimin8Only = rank
                break
        
        rows.append({
            "sample_name": sample_name,
            "gene": genes,
            "variant": variant,
            "type": answer["type"],
            # DAVP-full
            "davp_gene_rank": gene_rank_full,
            "davp_variant_rank": variant_rank_full,
            # DAVP-noTournament
            "davp_noTournament_gene_rank": gene_rank_noTournament,
            "davp_noTournament_variant_rank": variant_rank_noTournament,
            # DAVP-prelimin8Only
            "davp_prelimin8Only_gene_rank": gene_rank_prelimin8Only,
            "davp_prelimin8Only_variant_rank": None
        })
    
    return pd.DataFrame(rows)


def calculate_recall(df: pd.DataFrame, rank_col: str, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, float]:
    """Calculate Top-K recall metrics"""
    valid = df[rank_col].dropna()
    n_total = len(df)
    n_valid = len(valid)
    metrics = {"N": n_total, "Found": n_valid}
    for k in k_values:
        count = (valid <= k).sum()
        # Calculate recall as percentage of samples WITH results, not total benchmark
        metrics[f"Top-{k}"] = (count / n_valid) * 100 if n_valid > 0 else 0.0
    return metrics


def plot_recall_bars(df: pd.DataFrame, method_names: List[str], rank_type: str = 'gene', 
                     title: str = '', output_path: Path = None):
    """Plot Top-K recall as grouped bar chart"""
    k_values = [1, 3, 5, 10, 20]
    
    # Calculate recall for each method
    method_data = {}
    for method in method_names:
        col_name = f"{method}_{rank_type}_rank"
        if col_name not in df.columns:
            continue
        
        recall_metrics = calculate_recall(df, col_name, k_values)
        method_data[method] = [recall_metrics.get(f"Top-{k}", 0) for k in k_values]
    
    if not method_data:
        print(f"No data to plot for {rank_type} ranking")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(k_values))
    width = 0.25
    multiplier = 0
    
    method_labels = {
        'davp': 'DAVP-full',
        'davp_noTournament': 'DAVP-noTournament',
        'davp_prelimin8Only': 'DAVP-prelimin8Only'
    }
    
    for method, recalls in method_data.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, recalls, width, label=method_labels.get(method, method.upper()))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%',
                       ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    ax.set_xlabel('Top-K', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_path: Path):
    """Generate summary table with recall metrics"""
    methods = ['davp', 'davp_noTournament', 'davp_prelimin8Only']
    method_labels = {
        'davp': 'DAVP-full',
        'davp_noTournament': 'DAVP-noTournament',
        'davp_prelimin8Only': 'DAVP-prelimin8Only'
    }
    
    rows = []
    for method in methods:
        gene_stats = calculate_recall(df, f"{method}_gene_rank", [1, 3, 5, 10, 20])
        variant_stats = calculate_recall(df, f"{method}_variant_rank", [1, 3, 5, 10, 20])
        
        rows.append({
            "Method": method_labels[method],
            "Level": "Gene",
            **{k: v for k, v in gene_stats.items() if k.startswith("Top-")}
        })
        
        if method != 'davp_prelimin8Only':  # No variant ranking for prelimin8Only
            rows.append({
                "Method": method_labels[method],
                "Level": "Variant",
                **{k: v for k, v in variant_stats.items() if k.startswith("Top-")}
            })
    
    table_df = pd.DataFrame(rows)
    
    # Save as CSV
    table_df.to_csv(output_path, index=False, float_format="%.2f")
    print(f"Saved summary table to {output_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("ABLATION ANALYSIS SUMMARY")
    print("="*80)
    print(table_df.to_string(index=False))
    print("="*80 + "\n")


def main():
    print("="*80)
    print("DAVP ANALYSIS")
    print("="*80 + "\n")
    
    # Find available benchmark files
    if not BENCHMARKS_DIR.exists():
        print(f"Error: Benchmarks directory not found at {BENCHMARKS_DIR}")
        return
    
    benchmark_files = sorted(BENCHMARKS_DIR.glob("*.jsonl"))
    
    if not benchmark_files:
        print(f"Error: No benchmark files found in {BENCHMARKS_DIR}")
        return
    
    # Display available benchmarks
    print("Available benchmarks:")
    for idx, bf in enumerate(benchmark_files, 1):
        print(f"  {idx}. {bf.name}")
    
    # Prompt user to select
    while True:
        try:
            choice = input(f"\nSelect benchmark (1-{len(benchmark_files)}) or press Enter for default [{benchmark_files[0].name}]: ").strip()
            
            if choice == "":
                benchmark_path = benchmark_files[0]
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(benchmark_files):
                benchmark_path = benchmark_files[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(benchmark_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nAnalysis cancelled.")
            return
    
    print(f"\nUsing benchmark: {benchmark_path.name}")
    print("-"*80 + "\n")
    
    # Load benchmark
    answers = load_benchmark(benchmark_path)
    sample_names = [a["sample_name"] for a in answers]
    print(f"Loaded {len(answers)} samples from benchmark\n")
    
    # Load input variants
    print("Loading input variants...")
    input_variants = {}
    for sample_name in sample_names:
        df = load_input_variants(sample_name)
        if not df.empty:
            input_variants[sample_name] = df
    print(f"Loaded input variants for {len(input_variants)} samples\n")
    
    # Aggregate results from each stage
    prelimin8_results = aggregate_prelimin8_results(sample_names)
    elimin8_results = aggregate_elimin8_results(sample_names)
    tournament_results = aggregate_tournament_results(sample_names)
    
    print(f"Prelimin8: {len(prelimin8_results)} samples")
    print(f"Elimin8: {len(elimin8_results)} samples")
    print(f"Tournament: {len(tournament_results)} samples\n")
    
    # Convert variant rankings to gene rankings
    elimin8_gene_results = convert_to_gene_rankings(elimin8_results, input_variants, answers)
    tournament_gene_results = convert_to_gene_rankings(tournament_results, input_variants, answers)
    
    # Create ablation dataframe
    results_df = create_ablation_dataframe(
        answers, prelimin8_results, elimin8_results, tournament_results,
        elimin8_gene_results, tournament_gene_results
    )
    
    # Save results
    results_path = OUTPUT_DIR / "ablation_results.jsonl"
    results_df.to_json(results_path, orient="records", lines=True)
    print(f"Saved ablation results to {results_path}\n")
    
    # Generate summary table
    generate_summary_table(results_df, OUTPUT_DIR / "summary_table.csv")
    
    # Generate bar plots
    print("Generating plots...")
    plot_recall_bars(
        results_df,
        method_names=['davp', 'davp_noTournament', 'davp_prelimin8Only'],
        rank_type='gene',
        title='Gene Ranking Performance (Ablation Analysis)',
        output_path=OUTPUT_DIR / "gene_ranking_recall.png"
    )
    
    plot_recall_bars(
        results_df,
        method_names=['davp', 'davp_noTournament'],
        rank_type='variant',
        title='Variant Ranking Performance (Ablation Analysis)',
        output_path=OUTPUT_DIR / "variant_ranking_recall.png"
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
