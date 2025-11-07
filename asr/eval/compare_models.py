#!/usr/bin/env python3
"""
Compare two model evaluation results side-by-side

Usage:
    python compare_models.py \
        --result1 outputs/base-model/evaluation_results.json \
        --result2 outputs/finetuned-model/evaluation_results.json \
        --output comparison_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(results_path: str) -> Dict:
    """Load evaluation results"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_overall_metrics(results1: Dict, results2: Dict, name1: str, name2: str) -> str:
    """Generate overall metrics comparison"""
    
    wer1 = results1['wer']['wer']
    wer2 = results2['wer']['wer']
    cer1 = results1['cer']
    cer2 = results2['cer']
    
    wer_diff = wer2 - wer1
    cer_diff = cer2 - cer1
    
    wer_winner = "✅" if wer1 < wer2 else ("❌" if wer1 > wer2 else "➖")
    cer_winner = "✅" if cer1 < cer2 else ("❌" if cer1 > cer2 else "➖")
    
    report = f"""
## 📊 Overall Metrics Comparison

| Metric | {name1} | {name2} | Difference | Winner |
|--------|---------|---------|------------|--------|
| **WER** | {wer1:.2f}% | {wer2:.2f}% | {wer_diff:+.2f}% | {name1} {wer_winner} |
| **CER** | {cer1:.2f}% | {cer2:.2f}% | {cer_diff:+.2f}% | {name1} {cer_winner} |
| **Samples** | {results1['num_samples']} | {results2['num_samples']} | - | - |

### Error Breakdown

| Error Type | {name1} | {name2} | Difference |
|------------|---------|---------|------------|
| Substitutions | {results1['wer']['substitutions']} | {results2['wer']['substitutions']} | {results2['wer']['substitutions'] - results1['wer']['substitutions']:+d} |
| Insertions | {results1['wer']['insertions']} | {results2['wer']['insertions']} | {results2['wer']['insertions'] - results1['wer']['insertions']:+d} |
| Deletions | {results1['wer']['deletions']} | {results2['wer']['deletions']} | {results2['wer']['deletions'] - results1['wer']['deletions']:+d} |
| Hits | {results1['wer']['hits']} | {results2['wer']['hits']} | {results2['wer']['hits'] - results1['wer']['hits']:+d} |

"""
    return report


def find_worst_regressions(predictions1: List[Dict], predictions2: List[Dict], top_n: int = 20) -> List[Dict]:
    """Find samples where model2 performs significantly worse than model1"""
    
    regressions = []
    
    for pred1, pred2 in zip(predictions1, predictions2):
        if pred1['audio_path'] != pred2['audio_path']:
            continue  # Skip if not matching
        
        wer1 = pred1.get('wer', 0) or 0
        wer2 = pred2.get('wer', 0) or 0
        wer_diff = wer2 - wer1
        
        if wer_diff > 0:  # Model2 worse
            regressions.append({
                'audio_path': pred1['audio_path'],
                'reference': pred1['reference'],
                'hypothesis1': pred1['hypothesis'],
                'hypothesis2': pred2['hypothesis'],
                'wer1': wer1,
                'wer2': wer2,
                'wer_diff': wer_diff,
                'audio_duration': pred1.get('audio_duration', 0)
            })
    
    # Sort by WER difference (worst regressions first)
    regressions.sort(key=lambda x: x['wer_diff'], reverse=True)
    
    return regressions[:top_n]


def find_best_improvements(predictions1: List[Dict], predictions2: List[Dict], top_n: int = 20) -> List[Dict]:
    """Find samples where model2 performs significantly better than model1"""
    
    improvements = []
    
    for pred1, pred2 in zip(predictions1, predictions2):
        if pred1['audio_path'] != pred2['audio_path']:
            continue
        
        wer1 = pred1.get('wer', 0) or 0
        wer2 = pred2.get('wer', 0) or 0
        wer_diff = wer1 - wer2  # Positive means improvement
        
        if wer_diff > 0:  # Model2 better
            improvements.append({
                'audio_path': pred1['audio_path'],
                'reference': pred1['reference'],
                'hypothesis1': pred1['hypothesis'],
                'hypothesis2': pred2['hypothesis'],
                'wer1': wer1,
                'wer2': wer2,
                'wer_improvement': wer_diff,
                'audio_duration': pred1.get('audio_duration', 0)
            })
    
    # Sort by WER improvement (best improvements first)
    improvements.sort(key=lambda x: x['wer_improvement'], reverse=True)
    
    return improvements[:top_n]


def generate_regression_report(regressions: List[Dict], name1: str, name2: str) -> str:
    """Generate report for worst regressions"""
    
    if not regressions:
        return f"\n## ✅ No Regressions!\n\n{name2} performs at least as well as {name1} on all samples.\n"
    
    report = f"\n## ⚠️ Top {len(regressions)} Worst Regressions ({name2} worse than {name1})\n\n"
    
    for i, reg in enumerate(regressions, 1):
        report += f"### {i}. {Path(reg['audio_path']).name} (WER: {reg['wer1']:.1f}% → {reg['wer2']:.1f}%, +{reg['wer_diff']:.1f}%)\n\n"
        report += f"**Reference:** `{reg['reference']}`\n\n"
        report += f"**{name1}:** `{reg['hypothesis1']}`\n\n"
        report += f"**{name2}:** `{reg['hypothesis2']}`\n\n"
        report += "---\n\n"
    
    return report


def generate_improvement_report(improvements: List[Dict], name1: str, name2: str) -> str:
    """Generate report for best improvements"""
    
    if not improvements:
        return f"\n## ❌ No Improvements\n\n{name2} does not improve on any samples compared to {name1}.\n"
    
    report = f"\n## ✨ Top {len(improvements)} Best Improvements ({name2} better than {name1})\n\n"
    
    for i, imp in enumerate(improvements, 1):
        report += f"### {i}. {Path(imp['audio_path']).name} (WER: {imp['wer1']:.1f}% → {imp['wer2']:.1f}%, -{imp['wer_improvement']:.1f}%)\n\n"
        report += f"**Reference:** `{imp['reference']}`\n\n"
        report += f"**{name1}:** `{imp['hypothesis1']}`\n\n"
        report += f"**{name2}:** `{imp['hypothesis2']}`\n\n"
        report += "---\n\n"
    
    return report


def generate_csv_comparison(predictions1: List[Dict], predictions2: List[Dict], output_path: str, name1: str, name2: str):
    """Generate CSV with side-by-side comparison"""
    
    rows = []
    
    for pred1, pred2 in zip(predictions1, predictions2):
        if pred1['audio_path'] != pred2['audio_path']:
            continue
        
        wer1 = pred1.get('wer', 0) or 0
        wer2 = pred2.get('wer', 0) or 0
        wer_diff = wer2 - wer1
        
        rows.append({
            'audio_path': pred1['audio_path'],
            'reference': pred1['reference'],
            f'{name1}_hypothesis': pred1['hypothesis'],
            f'{name2}_hypothesis': pred2['hypothesis'],
            f'{name1}_wer': wer1,
            f'{name2}_wer': wer2,
            'wer_diff': wer_diff,
            f'{name1}_cer': pred1.get('cer', 0),
            f'{name2}_cer': pred2.get('cer', 0),
            'audio_duration': pred1.get('audio_duration', 0),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved detailed CSV comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two ASR model evaluation results"
    )
    parser.add_argument("--result1", required=True, help="First model results (evaluation_results.json)")
    parser.add_argument("--result2", required=True, help="Second model results (evaluation_results.json)")
    parser.add_argument("--name1", default="Model 1", help="Name for first model")
    parser.add_argument("--name2", default="Model 2", help="Name for second model")
    parser.add_argument("--output", default="model_comparison.md", help="Output markdown file")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top examples to show")
    
    args = parser.parse_args()
    
    print(f"Loading results...")
    results1 = load_results(args.result1)
    results2 = load_results(args.result2)
    
    predictions1 = results1['predictions']
    predictions2 = results2['predictions']
    
    print(f"✓ Loaded {len(predictions1)} predictions from {args.result1}")
    print(f"✓ Loaded {len(predictions2)} predictions from {args.result2}")
    
    # Generate report
    print(f"\nGenerating comparison report...")
    
    report = f"# Model Comparison Report\n\n"
    report += f"**{args.name1}:** `{results1.get('model', 'Unknown')}`\n\n"
    report += f"**{args.name2}:** `{results2.get('model', 'Unknown')}`\n\n"
    report += f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n"
    
    # Overall metrics
    report += compare_overall_metrics(results1, results2, args.name1, args.name2)
    
    # Find regressions and improvements
    print(f"Finding worst regressions...")
    regressions = find_worst_regressions(predictions1, predictions2, args.top_n)
    
    print(f"Finding best improvements...")
    improvements = find_best_improvements(predictions1, predictions2, args.top_n)
    
    # Add to report
    report += generate_regression_report(regressions, args.name1, args.name2)
    report += generate_improvement_report(improvements, args.name1, args.name2)
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Saved comparison report to: {args.output}")
    
    # Generate CSV
    csv_path = Path(args.output).with_suffix('.csv')
    generate_csv_comparison(predictions1, predictions2, str(csv_path), args.name1, args.name2)
    
    print(f"\n📊 Summary:")
    print(f"  - Total samples: {len(predictions1)}")
    print(f"  - Regressions ({args.name2} worse): {len(regressions)}")
    print(f"  - Improvements ({args.name2} better): {len(improvements)}")
    print(f"  - WER difference: {results2['wer']['wer'] - results1['wer']['wer']:+.2f}%")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()

