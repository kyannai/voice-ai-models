#!/usr/bin/env python3
"""
Quick analysis script for evaluation results with per-sample metrics
Run this after calculate_metrics.py to get detailed insights
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_results(results_file: Path) -> pd.DataFrame:
    """Load evaluation results into DataFrame"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['predictions'])
    return df, data


def print_summary(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("DETAILED ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total samples: {len(df)}")
    if 'audio_duration' in df.columns:
    print(f"  Total audio duration: {df['audio_duration'].sum():.1f}s ({df['audio_duration'].sum()/60:.1f} min)")
    if 'total_words' in df.columns:
    print(f"  Total words: {df['total_words'].sum()}")
    
    print(f"\n✅ Perfect Transcriptions (WER = 0%):")
    perfect = df[df['wer'] == 0.0]
    print(f"  Count: {len(perfect)} ({len(perfect)/len(df)*100:.1f}%)")
    
    print(f"\n📈 WER Statistics:")
    print(f"  Mean: {df['wer'].mean():.2f}%")
    print(f"  Median: {df['wer'].median():.2f}%")
    print(f"  Std Dev: {df['wer'].std():.2f}%")
    print(f"  Min: {df['wer'].min():.2f}%")
    print(f"  Max: {df['wer'].max():.2f}%")
    
    print(f"\n📈 CER Statistics:")
    print(f"  Mean: {df['cer'].mean():.2f}%")
    print(f"  Median: {df['cer'].median():.2f}%")
    
    if 'mer' in df.columns:
        print(f"\n📈 MER Statistics:")
        print(f"  Mean: {df['mer'].mean():.2f}%")
        print(f"  Median: {df['mer'].median():.2f}%")
    
    print(f"\n🎯 Performance Buckets:")
    excellent = df[df['wer'] < 5]
    good = df[(df['wer'] >= 5) & (df['wer'] < 10)]
    fair = df[(df['wer'] >= 10) & (df['wer'] < 20)]
    poor = df[df['wer'] >= 20]
    
    print(f"  WER < 5% (Excellent):  {len(excellent):4d} ({len(excellent)/len(df)*100:5.1f}%)")
    print(f"  WER 5-10% (Good):      {len(good):4d} ({len(good)/len(df)*100:5.1f}%)")
    print(f"  WER 10-20% (Fair):     {len(fair):4d} ({len(fair)/len(df)*100:5.1f}%)")
    print(f"  WER >= 20% (Poor):     {len(poor):4d} ({len(poor)/len(df)*100:5.1f}%)")
    
    # Error breakdown (only if available)
    if all(col in df.columns for col in ['substitutions', 'insertions', 'deletions']):
    print(f"\n🔍 Error Type Analysis:")
    total_errors = df['substitutions'].sum() + df['insertions'].sum() + df['deletions'].sum()
    if total_errors > 0:
        print(f"  Total errors: {total_errors}")
        print(f"  Substitutions: {df['substitutions'].sum():5d} ({df['substitutions'].sum()/total_errors*100:5.1f}%)")
        print(f"  Insertions:    {df['insertions'].sum():5d} ({df['insertions'].sum()/total_errors*100:5.1f}%)")
        print(f"  Deletions:     {df['deletions'].sum():5d} ({df['deletions'].sum()/total_errors*100:5.1f}%)")
    
    print(f"\n⚡ Performance Metrics:")
    if 'rtf' in df.columns:
    print(f"  Average RTF: {df['rtf'].mean():.3f}")
    if 'processing_time' in df.columns:
    print(f"  Total processing time: {df['processing_time'].sum():.1f}s ({df['processing_time'].sum()/60:.1f} min)")
    
    print("="*70)


def show_worst_samples(df: pd.DataFrame, n: int = 10):
    """Show samples with highest WER"""
    print(f"\n{'='*70}")
    print(f"TOP {n} SAMPLES WITH HIGHEST WER")
    print('='*70)
    
    worst = df.nlargest(n, 'wer')
    
    for idx, (_, row) in enumerate(worst.iterrows(), 1):
        audio_file = Path(row['audio_path']).name
        print(f"\n{idx}. {audio_file}")
        
        # Build metrics line
        metrics = [f"WER: {row['wer']:.2f}%", f"CER: {row['cer']:.2f}%"]
        if 'mer' in row:
            metrics.append(f"MER: {row['mer']:.2f}%")
        if 'rtf' in row:
            metrics.append(f"RTF: {row['rtf']:.3f}")
        print(f"   {' | '.join(metrics)}")
        
        # Show error breakdown if available
        if all(col in row for col in ['substitutions', 'insertions', 'deletions', 'total_words']):
        print(f"   Errors: {row['substitutions']}S {row['insertions']}I {row['deletions']}D | Words: {row['total_words']}")
        
        print(f"   REF: {row['reference']}")
        print(f"   HYP: {row['hypothesis']}")


def show_best_samples(df: pd.DataFrame, n: int = 5):
    """Show perfect or near-perfect samples"""
    print(f"\n{'='*70}")
    print(f"PERFECT/NEAR-PERFECT SAMPLES (showing {n})")
    print('='*70)
    
    best = df.nsmallest(n, 'wer')
    
    for idx, (_, row) in enumerate(best.iterrows(), 1):
        audio_file = Path(row['audio_path']).name
        print(f"\n{idx}. {audio_file}")
        print(f"   WER: {row['wer']:.2f}% | CER: {row['cer']:.2f}% | RTF: {row['rtf']:.3f}")
        print(f"   REF: {row['reference']}")
        print(f"   HYP: {row['hypothesis']}")


def export_analysis(df: pd.DataFrame, output_dir: Path):
    """Export detailed analysis files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add WER category
    def categorize_wer(wer):
        if wer == 0:
            return 'Perfect'
        elif wer < 5:
            return 'Excellent'
        elif wer < 10:
            return 'Good'
        elif wer < 20:
            return 'Fair'
        else:
            return 'Poor'
    
    df['wer_category'] = df['wer'].apply(categorize_wer)
    
    # Export full analysis
    analysis_file = output_dir / "detailed_analysis.csv"
    df.to_csv(analysis_file, index=False)
    print(f"\n✓ Saved detailed analysis: {analysis_file}")
    
    # Export high-error samples
    high_error = df[df['wer'] > 15.0].sort_values('wer', ascending=False)
    if len(high_error) > 0:
        high_error_file = output_dir / "high_error_samples.csv"
        high_error.to_csv(high_error_file, index=False)
        print(f"✓ Saved {len(high_error)} high-error samples: {high_error_file}")
    
    # Export perfect samples
    perfect = df[df['wer'] == 0.0]
    if len(perfect) > 0:
        perfect_file = output_dir / "perfect_samples.csv"
        perfect.to_csv(perfect_file, index=False)
        print(f"✓ Saved {len(perfect)} perfect samples: {perfect_file}")
    
    # Export summary stats
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("EVALUATION ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Perfect transcriptions: {len(perfect)} ({len(perfect)/len(df)*100:.1f}%)\n")
        f.write(f"Mean WER: {df['wer'].mean():.2f}%\n")
        f.write(f"Median WER: {df['wer'].median():.2f}%\n")
        f.write(f"Mean CER: {df['cer'].mean():.2f}%\n")
        if 'mer' in df.columns:
            f.write(f"Mean MER: {df['mer'].mean():.2f}%\n")
        if 'rtf' in df.columns:
        f.write(f"Average RTF: {df['rtf'].mean():.3f}\n")
    print(f"✓ Saved summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results with per-sample metrics"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="./results/evaluation_results.json",
        help="Path to evaluation_results.json file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis files (default: same as results)"
    )
    
    parser.add_argument(
        "--show-worst",
        type=int,
        default=10,
        help="Number of worst samples to show (default: 10)"
    )
    
    parser.add_argument(
        "--show-best",
        type=int,
        default=5,
        help="Number of best samples to show (default: 5)"
    )
    
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export analysis files"
    )
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    print(f"Loading results from: {results_file}")
    df, data = load_results(results_file)
    
    # Print summary
    print_summary(df)
    
    # Show worst samples
    if args.show_worst > 0:
        show_worst_samples(df, args.show_worst)
    
    # Show best samples
    if args.show_best > 0:
        show_best_samples(df, args.show_best)
    
    # Export analysis
    if not args.no_export:
        output_dir = args.output_dir if args.output_dir else results_file.parent
        export_analysis(df, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print('='*70)


if __name__ == "__main__":
    main()

