#!/usr/bin/env python3
"""
Analyze and visualize batch comparison results
Generate reports and identify interesting samples
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_comparison(comparison_file: str) -> dict:
    """Load comparison results from JSON"""
    with open(comparison_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_summary_csv(comparison: dict, output_file: str):
    """Generate CSV summary of overall metrics"""
    metrics = comparison["overall_metrics"]
    
    rows = []
    for model_key, model_metrics in metrics.items():
        rows.append({
            "model_key": model_key,
            "model": model_metrics["model"],
            "description": model_metrics["description"],
            "wer": model_metrics["wer"],
            "cer": model_metrics["cer"],
            "mer": model_metrics["mer"],
            "avg_rtf": model_metrics["avg_rtf"],
            "num_samples": model_metrics["num_samples"]
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("wer")  # Sort by WER
    df.to_csv(output_file, index=False)
    print(f"✓ Summary CSV saved to: {output_file}")


def generate_detailed_csv(comparison: dict, output_file: str):
    """Generate detailed CSV with all samples and hypotheses"""
    rows = []
    
    model_keys = list(comparison["overall_metrics"].keys())
    
    for sample in comparison["samples"]:
        row = {
            "sample_id": sample["sample_id"],
            "audio_path": sample["audio_path"],
            "reference": sample["reference"],
            "audio_duration": sample.get("audio_duration")
        }
        
        # Add hypothesis and metrics for each model
        for model_key in model_keys:
            row[f"{model_key}_hypothesis"] = sample["hypotheses"].get(model_key, "")
            row[f"{model_key}_wer"] = sample["metrics"].get(model_key, {}).get("wer")
            row[f"{model_key}_cer"] = sample["metrics"].get(model_key, {}).get("cer")
            row[f"{model_key}_mer"] = sample["metrics"].get(model_key, {}).get("mer")
            row[f"{model_key}_rtf"] = sample["metrics"].get(model_key, {}).get("rtf")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ Detailed CSV saved to: {output_file}")


def find_interesting_samples(comparison: dict, n: int = 20):
    """Find interesting samples where models disagree"""
    
    model_keys = list(comparison["overall_metrics"].keys())
    
    # Find samples with high variance in WER across models
    variances = []
    
    for sample in comparison["samples"]:
        wers = []
        for model_key in model_keys:
            wer = sample["metrics"].get(model_key, {}).get("wer")
            if wer is not None:
                wers.append(wer)
        
        if wers:
            variance = max(wers) - min(wers)  # WER range
            variances.append({
                "sample_id": sample["sample_id"],
                "audio_path": sample["audio_path"],
                "reference": sample["reference"],
                "wer_variance": variance,
                "min_wer": min(wers),
                "max_wer": max(wers),
                "hypotheses": sample["hypotheses"],
                "metrics": sample["metrics"]
            })
    
    # Sort by variance (descending)
    variances.sort(key=lambda x: x["wer_variance"], reverse=True)
    
    return variances[:n]


def print_interesting_samples(samples: list):
    """Print interesting samples where models disagree most"""
    print("\n" + "="*80)
    print(f"TOP {len(samples)} SAMPLES WITH HIGHEST MODEL DISAGREEMENT")
    print("="*80)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. {Path(sample['audio_path']).name}")
        print(f"   WER variance: {sample['wer_variance']:.1f}% (min: {sample['min_wer']:.1f}%, max: {sample['max_wer']:.1f}%)")
        print(f"   REF: {sample['reference']}")
        print()
        
        # Sort models by WER for this sample
        model_results = []
        for model_key, hypothesis in sample["hypotheses"].items():
            wer = sample["metrics"].get(model_key, {}).get("wer", 0)
            model_results.append((model_key, wer, hypothesis))
        
        model_results.sort(key=lambda x: x[1])  # Sort by WER
        
        for model_key, wer, hypothesis in model_results:
            print(f"   [{model_key:<25}] WER={wer:5.1f}% | {hypothesis}")
        
        print("-" * 80)


def generate_markdown_report(comparison: dict, interesting_samples: list, output_file: str):
    """Generate markdown report"""
    
    report = []
    report.append("# ASR Model Comparison Report\n")
    report.append(f"**Generated:** {comparison['metadata']['created']}\n")
    report.append(f"**Number of models:** {comparison['metadata']['num_models']}\n")
    report.append(f"**Number of samples:** {comparison['metadata']['num_samples']}\n")
    report.append("\n---\n")
    
    # Overall metrics table
    report.append("\n## Overall Metrics\n")
    report.append("\n| Model | WER | CER | MER | Avg RTF | Samples |\n")
    report.append("|-------|-----|-----|-----|---------|----------|\n")
    
    # Sort by WER
    sorted_metrics = sorted(
        comparison["overall_metrics"].items(),
        key=lambda x: x[1].get("wer", float('inf'))
    )
    
    for model_key, metrics in sorted_metrics:
        report.append(
            f"| {model_key} | "
            f"{metrics.get('wer', 0):.4f} | "
            f"{metrics.get('cer', 0):.4f} | "
            f"{metrics.get('mer', 0):.4f} | "
            f"{metrics.get('avg_rtf', 0):.3f} | "
            f"{metrics.get('num_samples', 0)} |\n"
        )
    
    # Best and worst performing
    if sorted_metrics:
        best_model = sorted_metrics[0]
        worst_model = sorted_metrics[-1]
        
        report.append(f"\n### 🏆 Best Model\n")
        report.append(f"**{best_model[0]}** - WER: {best_model[1].get('wer', 0):.4f}\n")
        
        report.append(f"\n### ⚠️ Worst Model\n")
        report.append(f"**{worst_model[0]}** - WER: {worst_model[1].get('wer', 0):.4f}\n")
    
    # Interesting samples
    report.append("\n---\n")
    report.append(f"\n## Top {len(interesting_samples)} Samples with Highest Disagreement\n")
    report.append("\nThese are samples where models produce significantly different results:\n")
    
    for i, sample in enumerate(interesting_samples, 1):
        report.append(f"\n### {i}. {Path(sample['audio_path']).name}\n")
        report.append(f"\n**Reference:** `{sample['reference']}`\n")
        report.append(f"\n**WER Variance:** {sample['wer_variance']:.1f}% (range: {sample['min_wer']:.1f}% - {sample['max_wer']:.1f}%)\n")
        report.append("\n**Hypotheses:**\n")
        
        # Sort by WER
        model_results = []
        for model_key, hypothesis in sample["hypotheses"].items():
            wer = sample["metrics"].get(model_key, {}).get("wer", 0)
            model_results.append((model_key, wer, hypothesis))
        
        model_results.sort(key=lambda x: x[1])
        
        for model_key, wer, hypothesis in model_results:
            report.append(f"- **{model_key}** (WER={wer:.1f}%): `{hypothesis}`\n")
        
        report.append("\n---\n")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✓ Markdown report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze batch comparison results"
    )
    
    parser.add_argument(
        "--comparison",
        type=str,
        default="outputs/batch_comparison.json",
        help="Path to comparison JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory for analysis files"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top disagreement samples to show (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Load comparison
    print(f"Loading comparison from: {args.comparison}")
    comparison = load_comparison(args.comparison)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary CSV
    summary_csv = output_dir / "summary_metrics.csv"
    generate_summary_csv(comparison, str(summary_csv))
    
    # Generate detailed CSV
    detailed_csv = output_dir / "detailed_comparison.csv"
    generate_detailed_csv(comparison, str(detailed_csv))
    
    # Find interesting samples
    print(f"\nFinding top {args.top_n} samples with highest disagreement...")
    interesting_samples = find_interesting_samples(comparison, args.top_n)
    
    # Print to console
    print_interesting_samples(interesting_samples)
    
    # Generate markdown report
    report_md = output_dir / "comparison_report.md"
    generate_markdown_report(comparison, interesting_samples, str(report_md))
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - {summary_csv.name}")
    print(f"  - {detailed_csv.name}")
    print(f"  - {report_md.name}")
    print("="*80)


if __name__ == "__main__":
    main()

