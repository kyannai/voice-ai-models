# Evaluation Methodology
# Malaysian Multilingual TTS System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** ML & QA Team

---

## 1. Evaluation Framework Overview

### 1.1 Evaluation Philosophy

Our evaluation approach is multi-dimensional, combining:

1. **Objective Metrics**: Quantitative measurements (MCD, F0, WER)
2. **Subjective Metrics**: Human perception (MOS, preference tests)
3. **Specialized Metrics**: Malaysian-specific (code-switching, particles)
4. **System Metrics**: Performance and reliability (latency, throughput)

### 1.2 Evaluation Stages

```
Stage 1: Development Evaluation
    ├─ Continuous during training
    ├─ Fast, automatic metrics
    └─ Used for model selection

Stage 2: Pre-Release Evaluation
    ├─ Comprehensive objective tests
    ├─ Initial subjective tests (10-15 raters)
    └─ Decision point for release

Stage 3: Production Evaluation
    ├─ Large-scale MOS (30+ raters)
    ├─ A/B testing
    ├─ User feedback collection
    └─ Continuous monitoring
```

---

## 2. Objective Metrics

### 2.1 Mel-Cepstral Distortion (MCD)

**Definition**: Spectral distance between generated and ground-truth audio

**Formula**:
```
MCD = (10 / ln(10)) × √(2 × Σ(c_gen[i] - c_target[i])²)
```

**Implementation**:
```python
# evaluation/metrics.py

from fastdtw import fastdtw
import librosa
import numpy as np

def compute_mcd(audio_gen, audio_target, sr=22050):
    """
    Compute Mel-Cepstral Distortion
    
    Args:
        audio_gen: Generated audio
        audio_target: Ground truth audio
        sr: Sample rate
    
    Returns:
        mcd: MCD value in dB
    """
    # Extract MFCCs
    mfcc_gen = librosa.feature.mfcc(y=audio_gen, sr=sr, n_mfcc=13)
    mfcc_target = librosa.feature.mfcc(y=audio_target, sr=sr, n_mfcc=13)
    
    # Remove 0th coefficient (energy)
    mfcc_gen = mfcc_gen[1:, :]
    mfcc_target = mfcc_target[1:, :]
    
    # Dynamic Time Warping alignment
    _, path = fastdtw(mfcc_gen.T, mfcc_target.T, dist='euclidean')
    path_gen = [p[0] for p in path]
    path_target = [p[1] for p in path]
    
    # Compute MCD
    mfcc_gen_aligned = mfcc_gen[:, path_gen]
    mfcc_target_aligned = mfcc_target[:, path_target]
    
    diff = mfcc_gen_aligned - mfcc_target_aligned
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2 * np.mean(np.sum(diff**2, axis=0)))
    
    return mcd
```

**Targets**:
- Excellent: < 5.5 dB
- Good: 5.5-6.5 dB  
- Acceptable: 6.5-8.0 dB
- Poor: > 8.0 dB

### 2.2 Pitch Metrics

#### 2.2.1 F0 Root Mean Square Error (RMSE)

```python
def compute_f0_rmse(audio_gen, audio_target, sr=22050):
    """
    Compute pitch RMSE
    """
    import pyworld as pw
    
    # Extract F0
    audio_gen = audio_gen.astype(np.float64)
    audio_target = audio_target.astype(np.float64)
    
    f0_gen, _ = pw.dio(audio_gen, sr)
    f0_gen = pw.stonemask(audio_gen, f0_gen, _, sr)
    
    f0_target, _ = pw.dio(audio_target, sr)
    f0_target = pw.stonemask(audio_target, f0_target, _, sr)
    
    # Align lengths
    min_len = min(len(f0_gen), len(f0_target))
    f0_gen = f0_gen[:min_len]
    f0_target = f0_target[:min_len]
    
    # Consider only voiced frames
    voiced_mask = (f0_gen > 0) & (f0_target > 0)
    
    if voiced_mask.sum() == 0:
        return float('inf')
    
    # RMSE
    rmse = np.sqrt(np.mean((f0_gen[voiced_mask] - f0_target[voiced_mask])**2))
    
    return rmse
```

**Targets**:
- Excellent: < 15 Hz
- Good: 15-25 Hz
- Acceptable: 25-40 Hz
- Poor: > 40 Hz

#### 2.2.2 F0 Correlation

```python
def compute_f0_correlation(audio_gen, audio_target, sr=22050):
    """
    Compute pitch correlation
    """
    # Extract F0 (same as above)
    # ...
    
    # Pearson correlation
    corr = np.corrcoef(f0_gen[voiced_mask], f0_target[voiced_mask])[0, 1]
    
    return corr
```

**Targets**:
- Excellent: > 0.90
- Good: 0.85-0.90
- Acceptable: 0.75-0.85
- Poor: < 0.75

#### 2.2.3 Gross Pitch Error (GPE)

```python
def compute_gpe(audio_gen, audio_target, sr=22050, threshold=20):
    """
    Compute Gross Pitch Error
    Percentage of frames with >20Hz error
    """
    # Extract F0 (same as above)
    # ...
    
    # Compute absolute error
    abs_error = np.abs(f0_gen[voiced_mask] - f0_target[voiced_mask])
    
    # Count gross errors
    gpe = (abs_error > threshold).sum() / len(abs_error) * 100
    
    return gpe
```

**Target**: < 10%

### 2.3 Duration Metrics

```python
def compute_duration_metrics(durations_pred, durations_target):
    """
    Compute duration prediction metrics
    
    Args:
        durations_pred: Predicted phoneme durations (frames)
        durations_target: Target durations (frames)
    
    Returns:
        mae: Mean Absolute Error (milliseconds)
        rmse: Root Mean Square Error (milliseconds)
    """
    hop_length = 256
    sr = 22050
    frame_to_ms = (hop_length / sr) * 1000
    
    mae = np.mean(np.abs(durations_pred - durations_target)) * frame_to_ms
    rmse = np.sqrt(np.mean((durations_pred - durations_target)**2)) * frame_to_ms
    
    return mae, rmse
```

**Targets**:
- MAE < 50ms
- RMSE < 70ms

### 2.4 Intelligibility: Word Error Rate (WER)

```python
def compute_wer_roundtrip(tts_model, asr_model, text):
    """
    ASR round-trip test for intelligibility
    
    Process:
    1. Generate audio from text (TTS)
    2. Transcribe audio (ASR)
    3. Compute WER between original and transcription
    """
    from jiwer import wer
    import whisper
    
    # Generate audio
    audio_gen = tts_model.synthesize(text)
    
    # Transcribe
    transcription = asr_model.transcribe(audio_gen)
    
    # Compute WER
    error_rate = wer(text, transcription)
    
    return error_rate, transcription
```

**Targets**:
- Excellent: < 3%
- Good: 3-5%
- Acceptable: 5-10%
- Poor: > 10%

### 2.5 Audio Quality Metrics

#### 2.5.1 Signal-to-Noise Ratio (SNR)

```python
def compute_snr(audio):
    """
    Estimate SNR of audio
    """
    # Estimate noise from silent segments
    energy = librosa.feature.rms(y=audio)[0]
    noise_threshold = np.percentile(energy, 10)
    
    noise_frames = energy < noise_threshold
    signal_frames = ~noise_frames
    
    if noise_frames.sum() == 0 or signal_frames.sum() == 0:
        return float('inf')
    
    signal_power = np.mean(audio[signal_frames]**2)
    noise_power = np.mean(audio[noise_frames]**2)
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db
```

**Target**: > 35 dB

#### 2.5.2 PESQ (Perceptual Evaluation of Speech Quality)

```python
from pesq import pesq

def compute_pesq(audio_gen, audio_target, sr=22050):
    """
    Compute PESQ score
    
    Note: PESQ requires 8kHz or 16kHz
    """
    # Resample to 16kHz for PESQ
    if sr != 16000:
        audio_gen_16k = librosa.resample(audio_gen, orig_sr=sr, target_sr=16000)
        audio_target_16k = librosa.resample(audio_target, orig_sr=sr, target_sr=16000)
    else:
        audio_gen_16k = audio_gen
        audio_target_16k = audio_target
    
    # Compute PESQ (wb = wideband mode)
    score = pesq(16000, audio_target_16k, audio_gen_16k, 'wb')
    
    return score
```

**Targets** (Wide-band):
- Excellent: > 4.0
- Good: 3.5-4.0
- Fair: 3.0-3.5
- Poor: < 3.0

#### 2.5.3 STOI (Short-Time Objective Intelligibility)

```python
from pystoi import stoi

def compute_stoi(audio_gen, audio_target, sr=22050):
    """
    Compute STOI score
    """
    score = stoi(audio_target, audio_gen, sr, extended=False)
    
    return score
```

**Targets**:
- Excellent: > 0.95
- Good: 0.90-0.95
- Fair: 0.80-0.90
- Poor: < 0.80

---

## 3. Subjective Metrics

### 3.1 Mean Opinion Score (MOS) - Naturalness

#### 3.1.1 Test Protocol

**Setup**:
- **Raters**: 30 native Malaysian speakers
- **Samples**: 50 utterances (randomized)
- **Rating Scale**: 1-5 (Likert scale)
- **Environment**: Quiet, headphones
- **Duration**: 20-30 minutes per rater

**Rating Scale**:
```
5 - Excellent: Completely natural, indistinguishable from human
4 - Good: Natural with minor imperfections
3 - Fair: Clearly synthetic but understandable
2 - Poor: Unnatural with significant artifacts
1 - Bad: Barely intelligible, very unnatural
```

**Sample Selection**:
- 15 pure Malay sentences
- 15 pure English sentences
- 15 code-switched sentences
- 5 particle-rich sentences

#### 3.1.2 MOS Test Interface

```html
<!-- mos_test.html -->

<div class="mos-test">
  <h2>Rate the naturalness of this speech:</h2>
  
  <div class="audio-player">
    <audio controls>
      <source src="sample_001.wav" type="audio/wav">
    </audio>
  </div>
  
  <div class="rating-scale">
    <button class="rating-btn" data-score="5">5 - Excellent</button>
    <button class="rating-btn" data-score="4">4 - Good</button>
    <button class="rating-btn" data-score="3">3 - Fair</button>
    <button class="rating-btn" data-score="2">2 - Poor</button>
    <button class="rating-btn" data-score="1">1 - Bad</button>
  </div>
  
  <div class="navigation">
    <button id="next-btn">Next →</button>
  </div>
  
  <div class="progress">
    Sample <span id="current">1</span> of <span id="total">50</span>
  </div>
</div>
```

#### 3.1.3 Statistical Analysis

```python
def analyze_mos_results(scores):
    """
    Analyze MOS test results
    
    Args:
        scores: List of (sample_id, rater_id, score) tuples
    
    Returns:
        analysis: Dictionary with statistics
    """
    import pandas as pd
    from scipy import stats
    
    df = pd.DataFrame(scores, columns=['sample_id', 'rater_id', 'score'])
    
    # Overall MOS
    mos_mean = df['score'].mean()
    mos_std = df['score'].std()
    mos_ci = stats.t.interval(
        0.95,
        len(df['score'])-1,
        loc=mos_mean,
        scale=stats.sem(df['score'])
    )
    
    # Per-sample MOS
    per_sample_mos = df.groupby('sample_id')['score'].agg(['mean', 'std', 'count'])
    
    # Per-rater statistics (to identify outliers)
    per_rater_stats = df.groupby('rater_id')['score'].agg(['mean', 'std'])
    
    # Inter-rater reliability (Cronbach's alpha)
    pivot = df.pivot(index='sample_id', columns='rater_id', values='score')
    alpha = compute_cronbach_alpha(pivot)
    
    analysis = {
        'overall_mos': mos_mean,
        'std': mos_std,
        'ci_95': mos_ci,
        'per_sample': per_sample_mos,
        'per_rater': per_rater_stats,
        'inter_rater_reliability': alpha
    }
    
    return analysis

def compute_cronbach_alpha(item_scores):
    """
    Compute Cronbach's alpha for inter-rater reliability
    """
    item_scores = item_scores.dropna()
    
    n_items = item_scores.shape[1]
    item_variances = item_scores.var(axis=0, ddof=1)
    total_variance = item_scores.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    
    return alpha
```

**Target**: MOS > 4.0 (with 95% CI)

### 3.2 MOS - Prosody

**Focus**: Rhythm, intonation, stress patterns

**Same protocol as MOS-Naturalness, but with different instructions:**

```
Rate the prosody (rhythm and intonation) of this speech:

5 - Excellent: Perfect rhythm, natural intonation
4 - Good: Mostly natural with minor issues
3 - Fair: Noticeable but not distracting
2 - Poor: Monotonous or unnatural rhythm
1 - Bad: Very poor prosody, difficult to listen to
```

**Target**: MOS-Prosody > 4.0

### 3.3 Code-Switching Quality Score (CSQS)

**Custom metric for Malaysian TTS**

#### 3.3.1 CSQS Questionnaire

For each code-switched sample, raters answer:

```
1. Language transitions are smooth and natural
   [1] Strongly Disagree ... [5] Strongly Agree

2. Pronunciation is correct in each language
   [1] Strongly Disagree ... [5] Strongly Agree

3. The accent sounds authentically Malaysian
   [1] Strongly Disagree ... [5] Strongly Agree

4. Overall, the code-switching sounds natural
   [1] Strongly Disagree ... [5] Strongly Agree
```

**CSQS = Average of 4 questions**

**Target**: CSQS > 4.0

### 3.4 Particle Quality Evaluation

**Specific test for particles**

#### 3.4.1 Test Design

**Samples**: 20 sentences with particles in different contexts

**Questions per sample**:
```
1. The particle sounds natural in this context
   [1] Strongly Disagree ... [5] Strongly Agree

2. The intonation of the particle is appropriate
   [1] Strongly Disagree ... [5] Strongly Agree

3. The emphasis/stress on the particle is correct
   [1] Strongly Disagree ... [5] Strongly Agree
```

**Particle Score = Average**

**Target**: Particle Quality > 4.2

### 3.5 Preference Tests (A/B Testing)

#### 3.5.1 Pairwise Comparison

```python
def conduct_preference_test(samples_a, samples_b, raters):
    """
    A/B preference test
    
    Args:
        samples_a: System A samples
        samples_b: System B samples (baseline or competitor)
        raters: List of rater IDs
    
    Returns:
        preference_results: Statistical results
    """
    results = []
    
    for i, (sample_a, sample_b) in enumerate(zip(samples_a, samples_b)):
        # Randomize order (blind test)
        if random.random() < 0.5:
            presented_order = (sample_a, sample_b, 'A')
        else:
            presented_order = (sample_b, sample_a, 'B')
        
        # Collect preference
        preference = get_rater_preference(presented_order, raters)
        
        results.append({
            'sample_id': i,
            'preference': preference,  # 'first', 'second', 'no_preference'
            'true_winner': presented_order[2]
        })
    
    # Analyze results
    analysis = analyze_preference_results(results)
    
    return analysis

def analyze_preference_results(results):
    """
    Statistical analysis of preference test
    """
    from scipy import stats
    
    df = pd.DataFrame(results)
    
    # Count preferences
    preference_counts = df['preference'].value_counts()
    
    # Binomial test (is preference significant?)
    n_total = len(df)
    n_preferred = preference_counts.get('first', 0)
    
    p_value = stats.binom_test(n_preferred, n_total, p=0.5, alternative='two-sided')
    
    # Effect size (Cliff's Delta)
    delta = (n_preferred - preference_counts.get('second', 0)) / n_total
    
    return {
        'preference_counts': preference_counts,
        'preference_rate': n_preferred / n_total,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': delta
    }
```

### 3.6 Rater Recruitment & Management

#### 3.6.1 Rater Qualifications

**Essential**:
- Native Malaysian speaker
- Age: 18-65
- Normal hearing
- Fluent in Malay and English

**Preferred**:
- Experience with audio evaluation
- Diverse age, gender, regional backgrounds
- No hearing aids or impairments

#### 3.6.2 Rater Training

**Training Session** (30 minutes):
1. Explanation of rating scales
2. Example samples with discussions
3. Practice rating (10 samples)
4. Calibration discussion
5. Q&A

#### 3.6.3 Quality Control

```python
def identify_outlier_raters(results):
    """
    Identify raters who consistently rate differently
    """
    df = pd.DataFrame(results)
    
    # Compute per-rater mean and std
    rater_stats = df.groupby('rater_id')['score'].agg(['mean', 'std'])
    
    # Overall mean and std
    overall_mean = df['score'].mean()
    overall_std = df['score'].std()
    
    # Flag raters more than 2 std away
    outliers = rater_stats[
        np.abs(rater_stats['mean'] - overall_mean) > 2 * overall_std
    ]
    
    return outliers.index.tolist()

def check_rater_consistency(results):
    """
    Check if raters are consistent (same sample rated twice)
    """
    # Include duplicate samples in test (hidden from raters)
    duplicates = find_duplicate_ratings(results)
    
    consistency_scores = []
    for rater_id, pairs in duplicates.items():
        diffs = [abs(r1 - r2) for r1, r2 in pairs]
        consistency = 1 - (np.mean(diffs) / 4)  # Normalize to 0-1
        consistency_scores.append((rater_id, consistency))
    
    return consistency_scores
```

---

## 4. Malaysian-Specific Evaluation

### 4.1 Language Detection Accuracy

```python
def evaluate_language_detection(tts_model, test_set):
    """
    Check if languages are correctly identified and pronounced
    """
    from langdetect import detect_langs
    
    results = []
    
    for sample in test_set:
        # Generate audio
        audio = tts_model.synthesize(sample['text'])
        
        # Transcribe with language detection
        transcription = asr_model.transcribe_with_lang_detection(audio)
        
        # Compare language sequences
        lang_accuracy = compare_language_sequences(
            sample['language_sequence'],
            transcription['language_sequence']
        )
        
        results.append({
            'sample_id': sample['id'],
            'language_accuracy': lang_accuracy,
            'correct_languages': sample['language_sequence'],
            'detected_languages': transcription['language_sequence']
        })
    
    avg_accuracy = np.mean([r['language_accuracy'] for r in results])
    
    return avg_accuracy, results
```

**Target**: > 95% language detection accuracy

### 4.2 Particle Intonation Analysis

```python
def analyze_particle_intonation(audio, particle_positions, particle_types):
    """
    Analyze if particle intonation is appropriate
    """
    import pyworld as pw
    
    # Extract pitch
    audio = audio.astype(np.float64)
    f0, timeaxis = pw.dio(audio, 22050)
    f0 = pw.stonemask(audio, f0, timeaxis, 22050)
    
    particle_analyses = []
    
    for pos, ptype in zip(particle_positions, particle_types):
        start_frame, end_frame = pos
        
        # Get particle pitch contour
        particle_f0 = f0[start_frame:end_frame]
        
        # Analyze contour
        if len(particle_f0) > 0:
            pitch_trend = np.polyfit(range(len(particle_f0)), particle_f0[particle_f0 > 0], 1)[0]
            pitch_mean = np.mean(particle_f0[particle_f0 > 0])
            
            # Expected patterns
            expected_patterns = {
                'emphatic': 'rising',  # lah
                'questioning': 'rising',  # meh, leh
                'resignation': 'falling',  # lor, loh
            }
            
            expected = expected_patterns.get(ptype, 'neutral')
            
            if expected == 'rising' and pitch_trend > 5:
                correct = True
            elif expected == 'falling' and pitch_trend < -5:
                correct = True
            elif expected == 'neutral':
                correct = abs(pitch_trend) < 5
            else:
                correct = False
            
            particle_analyses.append({
                'particle_type': ptype,
                'pitch_trend': pitch_trend,
                'expected': expected,
                'correct': correct
            })
    
    accuracy = sum(p['correct'] for p in particle_analyses) / len(particle_analyses)
    
    return accuracy, particle_analyses
```

**Target**: > 85% correct intonation patterns

### 4.3 Accent Authenticity Evaluation

**Qualitative assessment by native speakers**

**Test**: Present 10 samples and ask:

```
Question: Does this voice sound like it has a Malaysian accent?

Options:
- Definitely Malaysian
- Probably Malaysian
- Unsure
- Probably not Malaysian
- Definitely not Malaysian

Scoring: 5, 4, 3, 2, 1 respectively
```

**Target**: Mean score > 4.0

---

## 5. System Performance Metrics

### 5.1 Latency Metrics

```python
def measure_latency(tts_model, test_samples, num_runs=100):
    """
    Measure API latency
    """
    import time
    
    latencies = []
    
    for _ in range(num_runs):
        text = random.choice(test_samples)
        
        start = time.time()
        audio = tts_model.synthesize(text)
        end = time.time()
        
        latency = (end - start) * 1000  # milliseconds
        latencies.append(latency)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }
```

**Targets**:
- p50: < 200ms
- p95: < 500ms
- p99: < 1000ms

### 5.2 Real-Time Factor (RTF)

```python
def measure_rtf(tts_model, test_samples):
    """
    Measure Real-Time Factor
    
    RTF = synthesis_time / audio_duration
    RTF < 1.0 means faster than real-time
    """
    import time
    
    rtfs = []
    
    for text in test_samples:
        start = time.time()
        audio = tts_model.synthesize(text)
        synthesis_time = time.time() - start
        
        audio_duration = len(audio) / 22050  # seconds
        
        rtf = synthesis_time / audio_duration
        rtfs.append(rtf)
    
    return {
        'mean_rtf': np.mean(rtfs),
        'median_rtf': np.median(rtfs),
        'max_rtf': np.max(rtfs)
    }
```

**Target**: RTF < 0.3 (3x faster than real-time)

### 5.3 Throughput

```python
def measure_throughput(tts_model, num_requests=1000):
    """
    Measure requests per second
    """
    import concurrent.futures
    import time
    
    test_samples = generate_test_samples(num_requests)
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(tts_model.synthesize, text)
            for text in test_samples
        ]
        results = [f.result() for f in futures]
    
    elapsed = time.time() - start
    throughput = num_requests / elapsed
    
    return throughput  # requests per second
```

**Target**: > 100 requests/second (per GPU instance)

---

## 6. Evaluation Datasets

### 6.1 Test Set Design

**Composition**:
```python
test_set_composition = {
    # Language distribution
    'pure_malay': 100,
    'pure_english': 100,
    'pure_chinese': 50,
    'malay_english': 150,
    'malay_chinese': 50,
    'english_chinese': 50,
    'triple_mix': 50,
    
    # Special categories
    'particle_rich': 100,
    'long_sentences': 50,  # > 20 words
    'short_sentences': 50,  # < 5 words
    'numbers_dates': 30,
    'names_places': 30,
    'technical_terms': 30,
    'colloquial': 50,
    'formal': 50,
    
    # Edge cases
    'rare_words': 20,
    'homophones': 20,
    'tongue_twisters': 10,
    
    # Total: ~1000 samples
}
```

### 6.2 Benchmark Datasets

**Create standardized test sets for reproducibility**

```python
# scripts/create_benchmark.py

def create_malaysian_tts_benchmark():
    """
    Create standardized benchmark for Malaysian TTS
    """
    benchmark = {
        'metadata': {
            'name': 'Malaysian-TTS-Bench-v1.0',
            'version': '1.0',
            'date': '2025-10-12',
            'num_samples': 500,
            'languages': ['ms', 'en', 'zh'],
            'license': 'CC-BY-4.0'
        },
        'samples': []
    }
    
    # Add diverse samples
    benchmark['samples'].extend(load_pure_language_samples())
    benchmark['samples'].extend(load_code_switched_samples())
    benchmark['samples'].extend(load_particle_samples())
    benchmark['samples'].extend(load_edge_cases())
    
    # Save
    with open('benchmarks/malaysian_tts_bench_v1.json', 'w') as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    
    return benchmark
```

**Benefits**:
- Reproducible evaluations
- Easy comparison between models
- Standard reference for research
- Can be shared publicly

---

## 7. Evaluation Workflow

### 7.1 Automated Evaluation Pipeline

```python
# evaluation/evaluate.py

def run_full_evaluation(model_path, test_set_path, output_dir):
    """
    Run comprehensive automated evaluation
    """
    print("="*60)
    print("MALAYSIAN TTS EVALUATION")
    print("="*60)
    
    # Load model and test set
    model = load_model(model_path)
    test_set = load_test_set(test_set_path)
    
    results = {}
    
    # 1. Objective Metrics
    print("\n[1/5] Computing objective metrics...")
    results['objective'] = compute_objective_metrics(model, test_set)
    
    # 2. Intelligibility (WER)
    print("\n[2/5] Testing intelligibility (WER)...")
    results['wer'] = compute_wer_metrics(model, test_set)
    
    # 3. Malaysian-specific metrics
    print("\n[3/5] Evaluating code-switching and particles...")
    results['malaysian_specific'] = compute_malaysian_metrics(model, test_set)
    
    # 4. Performance metrics
    print("\n[4/5] Measuring performance (latency, RTF)...")
    results['performance'] = compute_performance_metrics(model, test_set)
    
    # 5. Generate samples for subjective testing
    print("\n[5/5] Generating samples for MOS testing...")
    generate_mos_samples(model, test_set, output_dir)
    
    # Save results
    save_results(results, output_dir)
    
    # Print summary
    print_evaluation_summary(results)
    
    return results

def print_evaluation_summary(results):
    """
    Print evaluation summary
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\nObjective Metrics:")
    print(f"  MCD:           {results['objective']['mcd']:.2f} dB")
    print(f"  F0 RMSE:       {results['objective']['f0_rmse']:.2f} Hz")
    print(f"  Duration MAE:  {results['objective']['duration_mae']:.2f} ms")
    print(f"  PESQ:          {results['objective']['pesq']:.2f}")
    print(f"  STOI:          {results['objective']['stoi']:.3f}")
    
    print("\nIntelligibility:")
    print(f"  WER:           {results['wer']['overall']:.2f}%")
    
    print("\nMalaysian-Specific:")
    print(f"  Lang Accuracy: {results['malaysian_specific']['lang_accuracy']:.1f}%")
    print(f"  Particle Score:{results['malaysian_specific']['particle_score']:.2f}")
    
    print("\nPerformance:")
    print(f"  Latency (p95): {results['performance']['latency_p95']:.0f} ms")
    print(f"  RTF:           {results['performance']['rtf']:.3f}")
    
    # Pass/Fail
    print("\n" + "="*60)
    passed = check_evaluation_criteria(results)
    if passed:
        print("✅ EVALUATION PASSED")
    else:
        print("❌ EVALUATION FAILED - Review metrics above")
    print("="*60 + "\n")
```

### 7.2 Subjective Evaluation Workflow

```
Step 1: Prepare Samples
    ├─ Generate audio for 50 test samples
    ├─ Include baselines/competitors
    └─ Randomize and anonymize

Step 2: Recruit Raters
    ├─ Post recruitment ad
    ├─ Screen applicants
    └─ Schedule sessions

Step 3: Training Session
    ├─ Explain rating scales
    ├─ Practice with examples
    └─ Calibration discussion

Step 4: Conduct Tests
    ├─ MOS-Naturalness (50 samples)
    ├─ MOS-Prosody (50 samples)
    ├─ CSQS (30 code-switched samples)
    └─ Particle Quality (20 samples)

Step 5: Analysis
    ├─ Statistical analysis
    ├─ Identify outliers
    └─ Generate report

Step 6: Decision
    ├─ Compare to targets
    └─ Release / iterate decision
```

---

## 8. Continuous Evaluation (Production)

### 8.1 Monitoring Metrics

```python
# monitoring/production_metrics.py

class ProductionMetrics:
    """
    Track metrics in production
    """
    def __init__(self):
        self.prometheus_client = init_prometheus()
    
    def log_synthesis_request(self, text, audio, latency, user_id):
        """
        Log each synthesis request
        """
        # Latency
        self.prometheus_client.histogram(
            'tts_latency_ms',
            latency * 1000
        )
        
        # RTF
        audio_duration = len(audio) / 22050
        rtf = latency / audio_duration
        self.prometheus_client.histogram('tts_rtf', rtf)
        
        # Text length
        self.prometheus_client.histogram('tts_text_length', len(text))
        
        # Audio quality (estimated)
        snr = estimate_snr(audio)
        self.prometheus_client.histogram('tts_audio_snr_db', snr)
    
    def log_user_feedback(self, synthesis_id, rating, comments):
        """
        Log user feedback
        """
        # Store in database for analysis
        db.insert_feedback({
            'synthesis_id': synthesis_id,
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now()
        })
```

### 8.2 A/B Testing Framework

```python
# evaluation/ab_testing.py

class ABTestManager:
    """
    Manage A/B tests for model improvements
    """
    def __init__(self):
        self.test_configs = load_test_configs()
        self.user_assignments = {}
    
    def assign_model_variant(self, user_id):
        """
        Assign user to model variant (A or B)
        """
        if user_id not in self.user_assignments:
            # Random assignment (50/50)
            variant = random.choice(['A', 'B'])
            self.user_assignments[user_id] = variant
        
        return self.user_assignments[user_id]
    
    def synthesize_with_variant(self, text, user_id):
        """
        Synthesize with assigned variant
        """
        variant = self.assign_model_variant(user_id)
        
        if variant == 'A':
            audio = model_a.synthesize(text)
        else:
            audio = model_b.synthesize(text)
        
        # Log for analysis
        log_synthesis(variant, text, audio, user_id)
        
        return audio
    
    def analyze_ab_test(self):
        """
        Analyze A/B test results
        """
        # Collect metrics for both variants
        metrics_a = get_metrics('A')
        metrics_b = get_metrics('B')
        
        # Statistical comparison
        results = {}
        for metric in ['latency', 'user_satisfaction', 'usage_duration']:
            t_stat, p_value = scipy.stats.ttest_ind(
                metrics_a[metric],
                metrics_b[metric]
            )
            
            results[metric] = {
                'variant_a_mean': np.mean(metrics_a[metric]),
                'variant_b_mean': np.mean(metrics_b[metric]),
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
```

---

## 9. Reporting

### 9.1 Evaluation Report Template

```markdown
# Malaysian TTS Evaluation Report

**Model Version**: v1.0  
**Evaluation Date**: 2025-10-12  
**Evaluator**: ML Team  

## Executive Summary

Overall assessment: [PASS/FAIL]
Key highlights: ...

## Objective Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MCD | 6.2 dB | < 6.5 dB | ✅ PASS |
| F0 RMSE | 22.5 Hz | < 25 Hz | ✅ PASS |
| WER | 4.3% | < 5% | ✅ PASS |
| PESQ | 3.8 | > 3.5 | ✅ PASS |

## Subjective Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MOS-Naturalness | 4.1 ± 0.3 | > 4.0 | ✅ PASS |
| MOS-Prosody | 3.9 ± 0.4 | > 4.0 | ⚠️ MARGINAL |
| CSQS | 4.2 ± 0.3 | > 4.0 | ✅ PASS |
| Particle Quality | 4.3 ± 0.3 | > 4.2 | ✅ PASS |

## Malaysian-Specific Evaluation

- Language Detection Accuracy: 96.2%
- Code-Switching Smoothness: 4.1/5
- Particle Intonation Accuracy: 87.3%

## Performance Metrics

- Latency (p95): 420ms (Target: < 500ms) ✅
- RTF: 0.28 (Target: < 0.3) ✅
- Throughput: 120 req/s (Target: > 100) ✅

## Detailed Analysis

### Strengths
- Excellent code-switching capability
- Particles sound natural
- Good intelligibility

### Weaknesses
- Prosody slightly below target
- Some rare words mispronounced
- Occasional artifacts in long sentences

### Recommendations
1. Fine-tune on prosody-focused dataset
2. Expand vocabulary coverage
3. Improve handling of long sentences

## Appendices

- Raw data
- Sample audio files
- Rater demographics
```

---

## 10. Conclusion

This evaluation methodology provides a comprehensive framework for assessing Malaysian TTS quality. Key principles:

1. **Multi-dimensional**: Objective + subjective + Malaysian-specific
2. **Rigorous**: Statistical analysis, multiple raters, reproducible
3. **Continuous**: From development through production
4. **Actionable**: Clear metrics and targets for decision-making

**Next Steps**:
1. Implement automated evaluation pipeline
2. Recruit and train MOS raters
3. Conduct baseline evaluation
4. Iterate based on results

---

**Document Version:** 1.0  
**Last Updated:** October 12, 2025

