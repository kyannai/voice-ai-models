# Evaluation Methodology
# Malaysian Multilingual ASR System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** ML Engineering & QA Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Objective Metrics](#2-objective-metrics)
3. [Malaysian-Specific Evaluation](#3-malaysian-specific-evaluation)
4. [Subjective Evaluation](#4-subjective-evaluation)
5. [Test Set Design](#5-test-set-design)
6. [Benchmarking](#6-benchmarking)
7. [Production Monitoring](#7-production-monitoring)
8. [Continuous Evaluation](#8-continuous-evaluation)

---

## 1. Overview

### 1.1 Evaluation Goals

**Primary Objectives:**
1. Measure transcription accuracy (WER, CER)
2. Assess code-switching detection quality
3. Evaluate discourse particle recognition
4. Measure inference speed and resource usage
5. Validate user satisfaction

**Success Criteria:**
- **Overall WER**: < 15% (target: < 12%)
- **Code-switching F1**: > 85%
- **Particle recall**: > 80%
- **User satisfaction**: > 4.0/5.0
- **Real-Time Factor (RTF)**: < 0.3

### 1.2 Evaluation Strategy

```
┌─────────────────────────────────────────────────────────┐
│                   EVALUATION PYRAMID                     │
│                                                          │
│                    ┌──────────────┐                     │
│                    │ Production   │                     │
│                    │ Monitoring   │  (Continuous)       │
│                    └──────────────┘                     │
│                ┌──────────────────────┐                 │
│                │ User Acceptance Test │ (Monthly)       │
│                └──────────────────────┘                 │
│            ┌────────────────────────────────┐           │
│            │ Subjective Evaluation (MOS)    │ (Weekly)  │
│            └────────────────────────────────┘           │
│        ┌──────────────────────────────────────────┐     │
│        │ Malaysian-Specific Metrics           │ (Daily) │
│        │ (Code-switching, Particles)          │         │
│        └──────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐ │
│    │ Objective Metrics (WER, CER, Latency)      │ (CI) │
│    └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Objective Metrics

### 2.1 Word Error Rate (WER)

**Definition:** Percentage of words incorrectly transcribed

**Formula:**
```
WER = (Substitutions + Insertions + Deletions) / Total Words × 100%
```

**Example:**
```
Reference: "Can you tolong check the system lah"
Hypothesis: "Can you too long check system"

Substitutions: "tolong" → "too long" (1)
Deletions: "the" (1), "lah" (1)
Total words: 7

WER = (1 + 0 + 2) / 7 = 42.9%
```

**Implementation:**
```python
import jiwer

def calculate_wer(reference: list[str], hypothesis: list[str]) -> float:
    """Calculate Word Error Rate."""
    wer = jiwer.wer(reference, hypothesis)
    return wer * 100  # Return as percentage

# Example
reference = ["Can you tolong check the system lah"]
hypothesis = ["Can you too long check system"]

wer = calculate_wer(reference, hypothesis)
print(f"WER: {wer:.2f}%")  # Output: WER: 42.86%
```

**Detailed Breakdown:**
```python
from jiwer import compute_measures

measures = compute_measures(reference, hypothesis)
print(f"Substitutions: {measures['substitutions']}")
print(f"Insertions: {measures['insertions']}")
print(f"Deletions: {measures['deletions']}")
print(f"Hits: {measures['hits']}")
```

**Target WER by Category:**

| Category | Target WER | World-Class WER |
|----------|-----------|-----------------|
| **Clean Speech** (SNR > 25dB) | < 12% | < 8% |
| **Mixed Code-Switching** | < 15% | < 12% |
| **Noisy Audio** (SNR 10-15dB) | < 22% | < 18% |
| **Call Center Audio** | < 18% | < 15% |
| **English-Only** (Malaysian accent) | < 8% | < 5% |
| **Malay-Only** | < 12% | < 8% |

### 2.2 Character Error Rate (CER)

**Definition:** Percentage of characters incorrectly transcribed

**Why CER?** More forgiving for morphological errors (e.g., "check" vs "checks")

**Formula:**
```
CER = (Char Substitutions + Insertions + Deletions) / Total Characters × 100%
```

**Implementation:**
```python
def calculate_cer(reference: list[str], hypothesis: list[str]) -> float:
    """Calculate Character Error Rate."""
    cer = jiwer.cer(reference, hypothesis)
    return cer * 100

# Example
reference = ["Can you tolong check"]
hypothesis = ["Can you too long checks"]

cer = calculate_cer(reference, hypothesis)
print(f"CER: {cer:.2f}%")  # Lower than WER for this example
```

**Target CER:**
- Overall: < 8%
- World-Class: < 5%

### 2.3 Sentence Error Rate (SER)

**Definition:** Percentage of sentences with at least one error

**Implementation:**
```python
def calculate_ser(reference: list[str], hypothesis: list[str]) -> float:
    """Calculate Sentence Error Rate."""
    errors = sum(1 for ref, hyp in zip(reference, hypothesis) if ref != hyp)
    return (errors / len(reference)) * 100

# Target: < 40% (i.e., 60%+ sentences perfect)
```

### 2.4 Real-Time Factor (RTF)

**Definition:** Ratio of processing time to audio duration

**Formula:**
```
RTF = Processing Time / Audio Duration
```

**Example:**
```
Audio duration: 60 seconds
Processing time: 18 seconds
RTF = 18 / 60 = 0.3
```

**Target RTF:**
- GPU (T4): < 0.3 (process 1 min audio in 18 sec)
- GPU (A100): < 0.15 (9 sec)
- CPU (16 cores): < 1.0 (60 sec, real-time)

**Implementation:**
```python
import time

def measure_rtf(audio_path: str, model):
    """Measure Real-Time Factor."""
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    
    # Measure processing time
    start_time = time.time()
    transcription = model.transcribe(audio_path)
    processing_time = time.time() - start_time
    
    rtf = processing_time / audio_duration
    
    return {
        "audio_duration": audio_duration,
        "processing_time": processing_time,
        "rtf": rtf,
    }

# Example
result = measure_rtf("test_audio.wav", model)
print(f"RTF: {result['rtf']:.3f}")
```

### 2.5 Throughput

**Definition:** Number of audio minutes processed per second

**Measurement:**
```python
import concurrent.futures
import time

def measure_throughput(audio_files: list[str], model, max_workers: int = 4):
    """Measure system throughput with concurrent requests."""
    
    start_time = time.time()
    total_duration = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for audio_file in audio_files:
            future = executor.submit(model.transcribe, audio_file)
            futures.append(future)
            
            # Calculate audio duration
            audio, sr = librosa.load(audio_file, sr=16000)
            total_duration += len(audio) / sr
        
        # Wait for all to complete
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    elapsed_time = time.time() - start_time
    throughput = total_duration / elapsed_time  # Minutes of audio per second
    
    return {
        "total_audio_duration": total_duration / 60,  # minutes
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "concurrent_requests": max_workers,
    }

# Target: > 2 min/sec (120x real-time) on T4 GPU with batching
```

---

## 3. Malaysian-Specific Evaluation

### 3.1 Code-Switching Detection

**Metric: F1-Score for Language Boundary Detection**

**Ground Truth Format:**
```json
{
  "text": "Can you tolong check the system",
  "words": [
    {"word": "Can", "language": "en"},
    {"word": "you", "language": "en"},
    {"word": "tolong", "language": "ms"},
    {"word": "check", "language": "en"},
    {"word": "the", "language": "en"},
    {"word": "system", "language": "en"}
  ]
}
```

**Evaluation Code:**
```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_code_switching(predictions: list, references: list):
    """
    Evaluate code-switching detection accuracy.
    
    Args:
        predictions: List of predicted language tags per word
        references: List of reference language tags per word
    
    Returns:
        Precision, Recall, F1-score for each language
    """
    
    # Flatten word-level predictions and references
    pred_flat = []
    ref_flat = []
    
    for pred, ref in zip(predictions, references):
        pred_flat.extend(pred["word_languages"])
        ref_flat.extend(ref["word_languages"])
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        ref_flat, pred_flat, labels=["en", "ms", "particle"], average=None
    )
    
    results = {}
    for i, lang in enumerate(["en", "ms", "particle"]):
        results[lang] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        }
    
    # Overall F1 (macro average)
    overall_f1 = f1.mean()
    
    return results, overall_f1

# Target: F1 > 0.85 overall
```

**Code-Switching Density:**
```python
def calculate_cs_density(text: str, word_languages: list[str]) -> float:
    """
    Calculate code-switching density (% of language switches).
    
    Example: "Can you tolong check" → 2 switches (en→ms, ms→en) / 4 words = 50%
    """
    switches = sum(
        1 for i in range(1, len(word_languages))
        if word_languages[i] != word_languages[i-1]
    )
    density = switches / len(word_languages) if len(word_languages) > 0 else 0
    return density

# Evaluate WER stratified by CS density
def wer_by_cs_density(test_set):
    """Report WER for low/medium/high code-switching density."""
    low_cs = [x for x in test_set if x["cs_density"] < 0.2]
    med_cs = [x for x in test_set if 0.2 <= x["cs_density"] < 0.5]
    high_cs = [x for x in test_set if x["cs_density"] >= 0.5]
    
    print(f"Low CS (< 20%): WER = {calculate_wer(low_cs)}%")
    print(f"Medium CS (20-50%): WER = {calculate_wer(med_cs)}%")
    print(f"High CS (> 50%): WER = {calculate_wer(high_cs)}%")
```

### 3.2 Particle Recognition

**Metric: Recall and Precision for Each Particle**

**Particle Inventory:**
```python
MALAYSIAN_PARTICLES = ["lah", "leh", "loh", "meh", "lor", "wor", "hor", "mah"]
```

**Evaluation:**
```python
def evaluate_particles(predictions: list[str], references: list[str]):
    """Evaluate discourse particle recognition."""
    
    results = {}
    
    for particle in MALAYSIAN_PARTICLES:
        # Count occurrences in references
        ref_count = sum(particle in ref.lower() for ref in references)
        
        # Count correctly predicted
        pred_count = sum(
            particle in pred.lower()
            for pred, ref in zip(predictions, references)
            if particle in ref.lower()
        )
        
        # Calculate recall
        recall = pred_count / ref_count if ref_count > 0 else 0
        
        # Count false positives
        fp_count = sum(
            particle in pred.lower()
            for pred, ref in zip(predictions, references)
            if particle not in ref.lower()
        )
        
        # Calculate precision
        total_pred = pred_count + fp_count
        precision = pred_count / total_pred if total_pred > 0 else 0
        
        results[particle] = {
            "recall": recall,
            "precision": precision,
            "f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            "occurrences": ref_count,
        }
    
    # Overall metrics
    overall_recall = sum(r["recall"] for r in results.values()) / len(MALAYSIAN_PARTICLES)
    overall_precision = sum(r["precision"] for r in results.values()) / len(MALAYSIAN_PARTICLES)
    
    return results, overall_recall, overall_precision

# Target: Recall > 0.80, Precision > 0.75
```

**Particle Confusion Matrix:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_particle_confusion(predictions, references):
    """Visualize which particles are confused with each other."""
    
    # Extract predicted and reference particles
    pred_particles = []
    ref_particles = []
    
    for pred, ref in zip(predictions, references):
        for particle in MALAYSIAN_PARTICLES:
            if particle in ref.lower():
                ref_particles.append(particle)
                # Find what was predicted instead
                found = False
                for p2 in MALAYSIAN_PARTICLES:
                    if p2 in pred.lower():
                        pred_particles.append(p2)
                        found = True
                        break
                if not found:
                    pred_particles.append("none")
    
    # Confusion matrix
    cm = confusion_matrix(ref_particles, pred_particles, labels=MALAYSIAN_PARTICLES + ["none"])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=MALAYSIAN_PARTICLES + ["none"], yticklabels=MALAYSIAN_PARTICLES)
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.title("Particle Confusion Matrix")
    plt.savefig("particle_confusion_matrix.png")
```

### 3.3 Accent & Domain Stratification

**Evaluate WER by Malaysian Sub-Dialect:**
```python
def wer_by_region(test_set):
    """Report WER for different Malaysian regional accents."""
    
    regions = {
        "kuala_lumpur": [x for x in test_set if x["accent"] == "kuala_lumpur"],
        "penang": [x for x in test_set if x["accent"] == "penang"],
        "johor": [x for x in test_set if x["accent"] == "johor"],
        "sabah_sarawak": [x for x in test_set if x["accent"] in ["sabah", "sarawak"]],
    }
    
    for region, samples in regions.items():
        if samples:
            wer = calculate_wer([s["reference"] for s in samples], [s["hypothesis"] for s in samples])
            print(f"{region}: WER = {wer:.2f}%")
```

**Evaluate by Domain:**
```python
def wer_by_domain(test_set):
    """Report WER for different domains."""
    
    domains = ["casual_conversation", "business", "call_center", "education", "news"]
    
    for domain in domains:
        samples = [x for x in test_set if x["domain"] == domain]
        if samples:
            wer = calculate_wer([s["reference"] for s in samples], [s["hypothesis"] for s in samples])
            print(f"{domain}: WER = {wer:.2f}%")
```

---

## 4. Subjective Evaluation

### 4.1 Mean Opinion Score (MOS)

**Definition:** Average rating by human evaluators

**Rating Scale (1-5):**
- **5 (Excellent)**: Perfect transcription, no errors
- **4 (Good)**: 1-2 minor errors, meaning clear
- **3 (Fair)**: 3-5 errors, meaning mostly clear
- **2 (Poor)**: 6-10 errors, meaning somewhat unclear
- **1 (Bad)**: > 10 errors or meaning lost

**Evaluation Protocol:**
```markdown
# MOS Evaluation Instructions

## Task
Listen to audio and read the transcription. Rate the transcription quality from 1-5.

## Rating Criteria
- **Accuracy**: Are the words correct?
- **Code-Switching**: Are language switches handled naturally?
- **Particles**: Are particles (lah, leh, loh) transcribed correctly?
- **Meaning**: Does the transcription preserve the speaker's intent?

## Example

Audio: [Play audio]
Transcription: "Can you tolong check the system lah"

Rate: [1] [2] [3] [4] [5]

Comments (optional): _______________________
```

**Sample Size:**
- Minimum: 100 samples per test condition
- Evaluators: 5-10 native Malaysian speakers
- Inter-rater reliability: Krippendorff's alpha > 0.7

**Implementation:**
```python
import numpy as np

def calculate_mos(ratings: list[list[int]]) -> dict:
    """
    Calculate Mean Opinion Score with confidence intervals.
    
    Args:
        ratings: List of ratings per sample. Each sample rated by multiple evaluators.
                 Example: [[5, 4, 5], [3, 4, 3], ...]  # 2 samples, 3 evaluators each
    
    Returns:
        MOS statistics (mean, std, confidence interval)
    """
    
    # Average ratings per sample
    sample_means = [np.mean(r) for r in ratings]
    
    # Overall MOS
    mos = np.mean(sample_means)
    std = np.std(sample_means)
    
    # 95% confidence interval
    ci = 1.96 * std / np.sqrt(len(sample_means))
    
    return {
        "mos": mos,
        "std": std,
        "ci_lower": mos - ci,
        "ci_upper": mos + ci,
        "n_samples": len(ratings),
        "n_evaluators": len(ratings[0]),
    }

# Target: MOS > 4.0
```

### 4.2 Preference Testing (A/B Test)

**Setup:**
- Compare your model (A) vs competitor (B) (e.g., Google ASR)
- Evaluators choose which transcription is better
- 50+ sample pairs per comparison

**Evaluation:**
```python
def preference_test(results: list[str]) -> dict:
    """
    Analyze A/B preference test results.
    
    Args:
        results: List of preferences. "A", "B", or "tie"
    
    Returns:
        Preference percentages and statistical significance
    """
    
    a_wins = results.count("A")
    b_wins = results.count("B")
    ties = results.count("tie")
    total = len(results)
    
    # Chi-square test for significance
    from scipy.stats import chisquare
    
    observed = [a_wins, b_wins, ties]
    expected = [total / 3, total / 3, total / 3]  # Null hypothesis: equal preference
    chi2, p_value = chisquare(observed, expected)
    
    return {
        "a_preference": a_wins / total,
        "b_preference": b_wins / total,
        "tie": ties / total,
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

# Target: A wins > 50% with p < 0.05
```

### 4.3 User Acceptance Testing (UAT)

**Procedure:**
1. Recruit 20-30 real users (content creators, call center managers, etc.)
2. Each user tests system with their own audio (10-20 minutes)
3. Collect feedback via survey

**Survey Questions:**
```markdown
# User Acceptance Survey

1. Overall transcription accuracy (1-5): ___
2. Code-switching handling (1-5): ___
3. Particle recognition (1-5): ___
4. Processing speed (1-5): ___
5. API ease of use (1-5): ___

6. Would you use this system in production? (Yes/No): ___
7. What is the biggest issue? (Free text): ___________
8. What feature do you want most? (Free text): _______

Net Promoter Score:
How likely are you to recommend this to a colleague? (0-10): ___
```

**Analysis:**
```python
def analyze_uat(responses: list[dict]) -> dict:
    """Analyze User Acceptance Test results."""
    
    # Average scores
    accuracy = np.mean([r["accuracy"] for r in responses])
    codeswitching = np.mean([r["codeswitching"] for r in responses])
    particles = np.mean([r["particles"] for r in responses])
    speed = np.mean([r["speed"] for r in responses])
    ease_of_use = np.mean([r["ease_of_use"] for r in responses])
    
    # Net Promoter Score (NPS)
    nps_scores = [r["nps"] for r in responses]
    promoters = sum(1 for s in nps_scores if s >= 9)
    detractors = sum(1 for s in nps_scores if s <= 6)
    nps = (promoters - detractors) / len(nps_scores) * 100
    
    # Adoption rate
    would_use = sum(1 for r in responses if r["would_use"] == "Yes")
    adoption_rate = would_use / len(responses)
    
    return {
        "accuracy_score": accuracy,
        "codeswitching_score": codeswitching,
        "particles_score": particles,
        "speed_score": speed,
        "ease_of_use_score": ease_of_use,
        "nps": nps,
        "adoption_rate": adoption_rate,
    }

# Target: All scores > 4.0, NPS > 50, adoption > 70%
```

---

## 5. Test Set Design

### 5.1 Test Set Requirements

**Size:**
- Minimum: 5 hours (500-1000 samples)
- Recommended: 10 hours (1000-2000 samples)

**Diversity:**

| Dimension | Distribution |
|-----------|--------------|
| **Language Mix** | 40% mixed, 30% English, 30% Malay |
| **Code-Switching Density** | 25% low, 50% medium, 25% high |
| **Particles** | All 8 particles represented (100+ each) |
| **Speakers** | 20+ unique speakers (balanced gender) |
| **Audio Quality** | 60% clean, 30% noisy, 10% very noisy |
| **Domains** | Casual (40%), business (30%), call center (20%), news (10%) |
| **Accents** | KL (40%), Penang (20%), Johor (20%), Sabah/Sarawak (20%) |

### 5.2 Test Set Curation

**Gold Standard Process:**
1. Professional transcribers create initial transcripts
2. Expert reviewers verify 100% of transcripts
3. Disagreements resolved by linguistic expert
4. Final transcripts triple-checked

**Quality Assurance:**
```python
def validate_test_set(test_set: list[dict]):
    """Validate test set meets requirements."""
    
    checks = {
        "size": len(test_set) >= 500,
        "speaker_diversity": len(set(x["speaker_id"] for x in test_set)) >= 20,
        "has_codeswitching": sum(1 for x in test_set if x["language"] == "mixed") / len(test_set) >= 0.3,
        "has_particles": all(
            sum(1 for x in test_set if particle in x["text"].lower()) >= 10
            for particle in MALAYSIAN_PARTICLES
        ),
    }
    
    print("Test Set Validation:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())
```

### 5.3 Stratified Evaluation

```python
def stratified_evaluation(model, test_set):
    """Evaluate model across multiple stratifications."""
    
    results = {}
    
    # Overall
    results["overall"] = calculate_wer(test_set)
    
    # By language mix
    for lang in ["mixed", "en", "ms"]:
        subset = [x for x in test_set if x["language"] == lang]
        results[f"language_{lang}"] = calculate_wer(subset)
    
    # By code-switching density
    for density in ["low", "medium", "high"]:
        subset = [x for x in test_set if x["cs_density_category"] == density]
        results[f"cs_{density}"] = calculate_wer(subset)
    
    # By audio quality
    for quality in ["clean", "noisy", "very_noisy"]:
        subset = [x for x in test_set if x["audio_quality"] == quality]
        results[f"quality_{quality}"] = calculate_wer(subset)
    
    # By domain
    for domain in ["casual", "business", "call_center", "news"]:
        subset = [x for x in test_set if x["domain"] == domain]
        results[f"domain_{domain}"] = calculate_wer(subset)
    
    return results
```

---

## 6. Benchmarking

### 6.1 Baseline Comparisons

**Competitors:**
1. OpenAI Whisper-large v3 (vanilla, no fine-tuning)
2. ElevenLabs (TTS/Voice AI leader, also offers ASR)
3. Google Cloud Speech-to-Text
4. AWS Transcribe
5. Azure Speech Services
6. AssemblyAI

**Benchmark Script:**
```python
import time

def benchmark_model(model_name: str, test_set: list[dict], model_api):
    """Benchmark a model on test set."""
    
    results = []
    total_time = 0
    total_audio_duration = 0
    
    for sample in test_set:
        audio_path = sample["audio_path"]
        reference = sample["text"]
        
        # Transcribe
        start_time = time.time()
        hypothesis = model_api.transcribe(audio_path)
        processing_time = time.time() - start_time
        
        # Calculate WER for this sample
        wer = calculate_wer([reference], [hypothesis])
        
        # Calculate audio duration
        audio_duration = sample["duration"]
        
        results.append({
            "sample_id": sample["id"],
            "wer": wer,
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "rtf": processing_time / audio_duration,
        })
        
        total_time += processing_time
        total_audio_duration += audio_duration
    
    # Aggregate metrics
    avg_wer = np.mean([r["wer"] for r in results])
    avg_rtf = total_time / total_audio_duration
    
    return {
        "model": model_name,
        "avg_wer": avg_wer,
        "avg_rtf": avg_rtf,
        "total_time": total_time,
        "total_audio": total_audio_duration,
        "sample_results": results,
    }

# Run benchmarks
models = [
    ("Our Model", our_model_api),
    ("Whisper-large v3", whisper_baseline_api),
    ("Google Cloud", google_api),
    # Add more...
]

benchmark_results = []
for model_name, model_api in models:
    print(f"Benchmarking {model_name}...")
    results = benchmark_model(model_name, test_set, model_api)
    benchmark_results.append(results)

# Compare
import pandas as pd
df = pd.DataFrame([
    {"Model": r["model"], "WER": f"{r['avg_wer']:.2f}%", "RTF": f"{r['avg_rtf']:.3f}"}
    for r in benchmark_results
])
print(df)
```

**Expected Results:**

| Model | WER (Overall) | WER (Code-Switching) | Particle Recall | RTF | Cost (/min) |
|-------|---------------|----------------------|-----------------|-----|-------------|
| **Our Model** | **13.5%** | **14.2%** | **82%** | **0.25** | **$0.006** |
| Whisper-large v3 (baseline) | 18.3% | 22.1% | 65% | 0.28 | $0.006 |
| ElevenLabs | 20.8% | 24.3% | 62% | 0.22 | $0.018 |
| Google Cloud | 24.5% | 28.7% | 58% | 0.15 | $0.024 |
| AWS Transcribe | 26.2% | 30.3% | 52% | 0.18 | $0.024 |
| Azure Speech | 25.8% | 29.5% | 55% | 0.16 | $0.018 |

**Note:** ElevenLabs is primarily a TTS (voice cloning) leader but also offers ASR. Their ASR quality is good for English but lacks Malaysian-specific optimizations.

---

## 7. Production Monitoring

### 7.1 Real-Time Metrics

**Dashboards (Grafana):**

```yaml
# prometheus_metrics.yml

- name: asr_requests_total
  type: counter
  help: "Total ASR requests"
  labels: [status, language]

- name: asr_wer_estimate
  type: gauge
  help: "Estimated WER on production samples"

- name: asr_processing_duration_seconds
  type: histogram
  help: "Processing time distribution"
  buckets: [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

- name: asr_audio_duration_seconds
  type: histogram
  help: "Audio duration distribution"

- name: asr_rtf
  type: histogram
  help: "Real-Time Factor distribution"
```

**Alert Rules:**
```yaml
# alert_rules.yml

- alert: HighWER
  expr: asr_wer_estimate > 0.20
  for: 1h
  annotations:
    summary: "WER increased above 20%"

- alert: SlowProcessing
  expr: asr_rtf > 0.5
  for: 15m
  annotations:
    summary: "RTF degraded above 0.5"

- alert: HighErrorRate
  expr: rate(asr_requests_total{status="error"}[5m]) > 0.05
  for: 10m
  annotations:
    summary: "Error rate above 5%"
```

### 7.2 Sample-Based Monitoring

**Random Sampling:**
```python
import random

def monitor_production_quality(transcriptions: list[dict], sample_rate: float = 0.01):
    """Sample and manually review production transcriptions."""
    
    # Random sample
    sample_size = int(len(transcriptions) * sample_rate)
    samples = random.sample(transcriptions, sample_size)
    
    # Queue for human review
    for sample in samples:
        send_to_review_queue(sample["audio_path"], sample["transcription"])
    
    # Collect feedback
    # (Human reviewers mark errors in UI)
    # Calculate estimated WER from sampled reviews
```

### 7.3 User Feedback Loop

**Feedback Collection:**
```python
# API response includes feedback link
{
  "transcription": "Can you tolong check the system lah",
  "confidence": 0.92,
  "feedback_url": "https://asr.example.com/feedback/txn_abc123"
}
```

**Feedback Form:**
```markdown
# Transcription Feedback

Audio ID: txn_abc123
Transcription: "Can you tolong check the system lah"

Is this transcription correct? [Yes] [No, see issues below]

If incorrect, please provide correct transcription:
_________________________________________________

What went wrong? (check all that apply)
[ ] Wrong words
[ ] Missing particle (lah, leh, etc.)
[ ] Wrong language detection
[ ] Gibberish
[ ] Other: _______________
```

---

## 8. Continuous Evaluation

### 8.1 Monthly Re-Evaluation

**Process:**
1. Collect 1000 new production samples
2. Send to professional transcribers
3. Create gold-standard test set
4. Re-evaluate model
5. Track WER drift over time

**Tracking:**
```python
import pandas as pd
import matplotlib.pyplot as plt

def track_wer_over_time(evaluation_results: list[dict]):
    """Plot WER trend over time."""
    
    df = pd.DataFrame(evaluation_results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["wer"], marker="o")
    plt.axhline(y=15, color="r", linestyle="--", label="Target WER (15%)")
    plt.xlabel("Date")
    plt.ylabel("WER (%)")
    plt.title("Model WER Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("wer_trend.png")

# Example data
evaluation_results = [
    {"date": "2025-10-01", "wer": 13.5},
    {"date": "2025-11-01", "wer": 13.8},
    {"date": "2025-12-01", "wer": 14.2},  # Slight degradation
]

track_wer_over_time(evaluation_results)
```

### 8.2 Automated Re-Training Triggers

**Trigger Conditions:**
```python
def should_retrain(current_wer: float, baseline_wer: float, threshold: float = 0.02):
    """Determine if model should be retrained based on WER drift."""
    
    drift = current_wer - baseline_wer
    
    if drift > threshold:
        return True, f"WER drifted by {drift:.2%} (threshold: {threshold:.2%})"
    
    return False, "Performance stable"

# Example
baseline_wer = 0.135  # 13.5%
current_wer = 0.158   # 15.8%

should_retrain, reason = should_retrain(current_wer, baseline_wer)
if should_retrain:
    print(f"Triggering re-training: {reason}")
    # Kick off automated re-training pipeline
```

### 8.3 A/B Testing New Models

**Gradual Rollout:**
```python
def route_request(user_id: str, model_a, model_b, rollout_percentage: int = 10):
    """Route traffic between old (A) and new (B) models."""
    
    # Hash user ID to ensure consistent routing
    import hashlib
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    
    # Route X% to model B, rest to model A
    if hash_value % 100 < rollout_percentage:
        return model_b
    else:
        return model_a

# Example: Gradually increase rollout if metrics look good
# 10% → 25% → 50% → 100%
```

---

## 9. Evaluation Checklist

### Before Launch
- [ ] WER < 15% on test set
- [ ] Code-switching F1 > 0.85
- [ ] Particle recall > 0.80
- [ ] MOS > 4.0 (from UAT)
- [ ] RTF < 0.3 on target hardware
- [ ] Benchmark against 3+ competitors
- [ ] Test set diversity validated
- [ ] 100+ user acceptance tests completed

### Post-Launch (Weekly)
- [ ] Monitor production WER estimate
- [ ] Review 1% sample of transcriptions
- [ ] Collect user feedback (NPS, surveys)
- [ ] Check for WER drift (> 2% = investigate)

### Monthly
- [ ] Re-evaluate on new test set (1000 samples)
- [ ] Update benchmark comparisons
- [ ] Analyze error patterns
- [ ] Decide on re-training

---

**End of Evaluation Methodology**

*For deployment procedures, see Deployment Guide (06).*

