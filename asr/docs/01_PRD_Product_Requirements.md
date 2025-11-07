# Product Requirements Document (PRD)
# Malaysian ASR System

---

## 1. Executive Summary

### 1.1 Product Vision

Create a production-grade Automatic Speech Recognition (ASR) system that accurately transcribes Malaysian speech patterns. The system is optimized for **Malay as the primary language** with natural English code-switching, Pinyin/Mandarin terms, local slang, and Malaysian discourse particles. Built on OpenAI's Whisper-large v3 and optimized with Unsloth for efficient fine-tuning.

### 1.2 Problem Statement

Current ASR solutions fail to handle the unique linguistic characteristics of Malaysian speech:

- **No Malay-primary code-switching**: Existing systems treat Malay and English as separate languages, failing to handle natural Malaysian speech where Malay is primary with English words mixed in (e.g., "Saya nak check system ni lah")
- **Poor particle recognition**: Malaysian discourse markers (lah, leh, loh, meh, lor) are frequently misrecognized or omitted (To be confirmed)
- **High error rates on mixed speech**: Whisper achieve high WER (To be found out) on Malaysian speech vs <10% on standard Malay or English alone
- **Wrong language detection**: Systems force single-language transcription, losing the natural Malay-English mix
- **Missing slang & special vocab**: Local terms, Pinyin words, and Malaysian-specific vocabulary not recognized
- **Unnatural transcriptions**: Output doesn't reflect how Malaysians actually speak, making it unusable

### 1.3 Competitive Landscape

| Provider | Strengths | Weaknesses | Est. MY WER | Pure Malay | Pure English | Source/Notes |
|----------|-----------|------------|-------------|------------|--------------|--------------|
| **Mesolitica/Malaysia-AI Whisper** | Malaysian-focused, open-source | Limited resources, smaller model | 15-20% | 12-15% | 10-13% | ⚠️ Estimated (requires benchmarking)⁵ |
| **ElevenLabs** | Best voice cloning, high quality | Premium pricing, ASR limited | 18-24% | 16-20% | 12-15% | ⚠️ Estimated (no public MY benchmarks) |
| **Google Cloud Speech** | Fast, 125+ languages | Poor code-switching | 22-28% | 15-20% | 10-14% | ⚠️ Estimated from multilingual benchmarks¹ |
| **AWS Transcribe** | Good infrastructure | No particle support | 25-32% | 18-22% | 12-16% | ⚠️ Estimated from SEA language performance² |
| **Azure Speech** | Enterprise features | Expensive, high WER | 24-30% | 17-22% | 12-16% | ⚠️ Estimated from Malay/English benchmarks³ |
| **Rev.ai** | Human-in-loop option | Manual, slow, costly | 15-20%* | 13-17% | 8-12% | Human transcription baseline |
| **AssemblyAI** | Good English | No Malaysian tuning | 20-26% | 18-24% | 8-12% | ⚠️ Estimated (English WER ~8%, code-switching degrades) |
| **Whisper-large v3** | Multilingual, open-source | Not Malaysian-tuned | 18-22% | 14-18% | 8-12% | Baseline for our fine-tuning⁴ |
| **Our Solution** | Malaysian-optimized | New entrant | **10-15%** ✓ | **8-12%** ✓ | **5-8%** ✓ | 🎯 **Target** (requires validation) |

**⚠️ IMPORTANT DISCLAIMERS:**
- **Estimates only**: Most WER figures are **estimated** based on general multilingual/code-switching performance
- **No official benchmarks**: None of these providers publish Malaysian-specific WER numbers
- **Validation needed**: Our 10-15% target requires real-world testing and validation
- **Scenario-dependent**: WER varies significantly based on:
  - Audio quality (clean studio vs noisy environment)
  - Code-switching frequency (high mixing degrades all systems)
  - Speaker accent variation
  - Domain (casual conversation vs formal speech)

*Rev.ai uses human transcribers for Malaysian content, not pure ASR

**Sources & Methodology (To be validated):**
1. **Google Cloud Speech**: Extrapolated from published multilingual benchmarks (8-12% WER on clean English, 15-20% on Malay). Code-switching typically adds 5-10% WER based on research literature ([Yilmaz et al., 2016](https://www.isca-speech.org/archive/interspeech_2016/yilmaz16_interspeech.html))
2. **AWS Transcribe**: Based on community reports and SEA language performance (no official MY benchmarks published)
3. **Azure Speech**: Estimated from published Malay (15-18%) and code-switching degradation (+6-12%)
4. **Whisper-large v3**: Vanilla model tested on Malaysian speech (internal testing required for actual numbers)
5. **Mesolitica/Malaysia-AI**: Open-source Malaysian Whisper fine-tuned models available on [HuggingFace](https://huggingface.co/malaysia-ai). Performance estimates based on fine-tuned Whisper models for Malaysian languages. Actual benchmarks needed for validation.


### 1.4 Success Criteria

**Must-Have (Launch Blockers):**
- Word Error Rate (WER) < 15% on Malaysian test set
- Code-switching detection accuracy > 85%
- Particle recognition accuracy > 80%
- API latency (p95) < 2 seconds for 1-minute audio

**Nice-to-Have (Post-Launch):**
- WER < 10% 
- Real-time streaming transcription support

---

## 2. Product Goals & Objectives

### 2.1 North Star Metrics

**Primary North Star:**
- **Monthly Transcription Hours**: 5,000 hours by Month 12

**Supporting Metrics:**
- Monthly Active Users (MAU): 1,000 by Month 12
- Daily Active API Calls: 3,000+ by Month 12
- Customer Satisfaction (CSAT): > 4.2/5.0

### 2.2 Product Objectives

**Quality Objectives (Objective Metrics):**
- **Overall WER**: < 15% (target: < 12%)
- **Character Error Rate (CER)**: < 8%
- **Code-switching accuracy**: > 85% (correct language per word)
- **Particle accuracy**: > 80% (lah, leh, loh, meh, lor)
- **English-only WER**: < 8%
- **Malay-only WER**: < 12%
- **Pinyin/Mandarin WER**: < 15%

**Performance Objectives:**
- **Real-Time Factor (RTF)**: < 0.3 (process 1 min audio in 18 seconds)
- **API latency (p50)**: < 800ms for 1-minute audio
- **API latency (p95)**: < 1.5s for 1-minute audio
- **API latency (p99)**: < 2.5s for 1-minute audio
- **System uptime**: > 99.5% (43 minutes downtime/month max)
- **Concurrent requests**: Support 500+ concurrent transcriptions

**User Experience Objectives:**
- **Transcription quality rating**: > 4.0/5.0 (user survey)
- **API ease of use**: > 4.5/5.0 (developer survey)
- **Documentation completeness**: > 4.3/5.0
- **Support response time**: < 4 hours (business hours)

---

## 3. Functional Requirements

### 3.1 Core ASR Functionality

#### FR-1: Audio Input Processing

**FR-1.1:** System SHALL accept audio files in the following formats:
- WAV (PCM, 16-bit, 8-48kHz)
- MP3 (32-320 kbps)
- M4A/AAC
- FLAC
- OGG/Vorbis

**FR-1.2:** System SHALL automatically resample audio to 16kHz mono for processing

**FR-1.3:** System SHALL support audio files from 1 second to 3 hours in duration

**FR-1.4:** System SHALL validate audio quality and reject files with:
- Bit depth < 8-bit
- Sample rate < 8kHz
- File size > 500MB
- Extreme noise (SNR < 5dB) with warning

**FR-1.5:** System SHALL accept audio input via:
- Direct file upload (multipart/form-data)
- URL (public or pre-signed)
- Base64-encoded audio data
- Streaming chunks (future: real-time)

---

#### FR-2: Multilingual Transcription

**FR-2.1:** System SHALL automatically detect and transcribe Malaysian code-switching speech containing:
- Malay (Bahasa Malaysia)
- English (Malaysian English)
- Mandarin Chinese (in Pinyin romanization)

**FR-2.2:** System SHALL preserve language boundaries and output mixed-language transcripts

**Example:**
```
Input Audio: "Okay so today kita akan discuss about the new features lah"
Output: "Okay so today kita akan discuss about the new features lah"
```

**FR-2.3:** System SHALL provide optional language tags per word/phrase:
```json
{
  "text": "Okay so today kita akan discuss about the new features lah",
  "words": [
    {"word": "Okay", "language": "en", "confidence": 0.95},
    {"word": "so", "language": "en", "confidence": 0.97},
    {"word": "today", "language": "en", "confidence": 0.96},
    {"word": "kita", "language": "ms", "confidence": 0.94},
    {"word": "akan", "language": "ms", "confidence": 0.92},
    {"word": "discuss", "language": "en", "confidence": 0.93},
    {"word": "about", "language": "en", "confidence": 0.96},
    {"word": "the", "language": "en", "confidence": 0.98},
    {"word": "new", "language": "en", "confidence": 0.97},
    {"word": "features", "language": "en", "confidence": 0.95},
    {"word": "lah", "language": "particle", "confidence": 0.89}
  ]
}
```

**FR-2.4:** System SHALL handle common Malaysian code-switching patterns:
- **Intra-sentential switching**: Within-sentence switches (most common)
- **Inter-sentential switching**: Between-sentence switches
- **Tag switching**: Discourse particles at sentence boundaries

---

#### FR-3: Discourse Particle Recognition

**FR-3.1:** System SHALL accurately recognize and transcribe Malaysian discourse particles:

| Particle | Function | Example |
|----------|----------|---------|
| **lah** | Emphasis, assertion | "Of course can lah!" |
| **leh** | Suggestion, softening | "We can try leh?" |
| **loh** | Obviousness, reminder | "I told you already loh!" |
| **meh** | Doubt, surprise | "Really meh?" |
| **lor** | Resignation, acceptance | "Cannot help it lor." |
| **wor** | Concern, uncertainty | "But how wor?" |
| **hor** | Confirmation seeking | "You know lah hor?" |
| **mah** | Explanation, obviousness | "He's busy mah!" |

**FR-3.2:** System SHALL distinguish particles from similar-sounding words:
- "lah" (particle) vs "la" (shortened "already")
- "leh" (particle) vs "ley" (name)
- "meh" (particle) vs "meh" (interjection of indifference)

**FR-3.3:** System SHALL preserve particle placement and intonation context

---

#### FR-4: Transcription Output Formats

**FR-4.1:** System SHALL provide transcriptions in multiple formats:

**Plain Text:**
```
Okay so today kita akan discuss about the new features lah. Then after that we boleh proceed to the next topic.
```

**JSON (Detailed):**
```json
{
  "text": "Okay so today kita akan discuss about the new features lah.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.5,
      "text": "Okay so today kita akan discuss about the new features lah.",
      "words": [...],
      "confidence": 0.92
    }
  ],
  "language": "mixed",
  "duration": 4.5
}
```

**SRT (Subtitles):**
```
1
00:00:00,000 --> 00:00:04,500
Okay so today kita akan discuss
about the new features lah.

2
00:00:04,500 --> 00:00:08,200
Then after that we boleh proceed
to the next topic.
```

**VTT (WebVTT):**
```
WEBVTT

00:00:00.000 --> 00:00:04.500
Okay so today kita akan discuss
about the new features lah.

00:00:04.500 --> 00:00:08.200
Then after that we boleh proceed
to the next topic.
```

**FR-4.2:** System SHALL provide word-level timestamps with ±0.5s accuracy

**FR-4.3:** System SHALL include confidence scores for:
- Overall transcription (0.0-1.0)
- Per-segment confidence
- Per-word confidence (optional)

---

#### FR-5: API Endpoints

**FR-5.1: POST /v1/transcribe** (Synchronous)
- Accepts audio file + parameters
- Returns complete transcription
- Max audio duration: 15 minutes
- Timeout: 60 seconds

**Request:**
```bash
curl -X POST https://api.asr.example.com/v1/transcribe \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@recording.mp3" \
  -F "language=mixed" \
  -F "output_format=json"
```

**Response:**
```json
{
  "id": "txn_abc123",
  "status": "completed",
  "text": "Okay so today kita akan discuss...",
  "duration": 45.2,
  "processing_time": 3.8,
  "confidence": 0.92
}
```

---

**FR-5.2: POST /v1/transcribe/async** (Asynchronous)
- Accepts audio file (up to 3 hours)
- Returns job ID immediately
- Webhook notification when complete

**Request:**
```bash
curl -X POST https://api.asr.example.com/v1/transcribe/async \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@long_recording.mp3" \
  -F "webhook_url=https://yourapp.com/webhook"
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "processing",
  "created_at": "2025-10-12T10:30:00Z",
  "estimated_completion": "2025-10-12T10:35:00Z"
}
```

---

**FR-5.3: GET /v1/transcribe/{job_id}** (Status Check)
- Returns current status of async job
- Includes transcription when complete

---

**FR-5.4: POST /v1/transcribe/stream** (Future: Real-time)
- WebSocket endpoint for streaming audio
- Returns partial transcriptions in real-time

---

### 3.2 Language & Localization

**FR-6: Language Support**

**FR-6.1:** System SHALL support the following language modes:
- `mixed` (default): Auto-detect Malaysian code-switching
- `en`: English-only transcription
- `ms`: Malay-only transcription
- `auto`: Auto-detect single language (fallback to mixed if multiple detected)

**FR-6.2:** System SHALL maintain consistent spelling conventions:
- Malay: Standard Bahasa Malaysia orthography
- English: Malaysian English variants (e.g., "favour" over "favor")
- Particles: Consistent romanization (lah, leh, loh)

**FR-6.3:** System SHALL handle common Malaysian abbreviations/slang:
- "kena" (affected by, must)
- "boleh" (can)
- "ada" (have/has)
- "mau" (want)
- "sudah/dah" (already)
- "sama" (same)
- "macam mana/camne" (how)

---

### 3.3 User Management & Authentication

**FR-7: API Authentication**

**FR-7.1:** System SHALL support API key-based authentication

**FR-7.2:** System SHALL support OAuth 2.0 for enterprise customers

**FR-7.3:** System SHALL enforce rate limiting per API key:
- Free tier: 10 requests/hour, 100 minutes/month
- Starter tier: 100 requests/hour, 1,000 minutes/month
- Pro tier: 1,000 requests/hour, 10,000 minutes/month
- Enterprise tier: Custom limits

**FR-7.4:** System SHALL provide API key management dashboard

---

### 3.4 Data Management

**FR-8: Audio Storage & Privacy**

**FR-8.1:** System SHALL allow users to opt-out of audio storage:
- Default: Audio deleted after 7 days
- Option: Delete immediately after transcription
- Option: Retain for 30 days (for debugging/re-processing)

**FR-8.2:** System SHALL encrypt all stored audio files (AES-256)

**FR-8.3:** System SHALL comply with Malaysia's Personal Data Protection Act (PDPA)

**FR-8.4:** System SHALL provide data export/deletion on user request (GDPR/PDPA compliance)

---

### 3.5 Monitoring & Analytics

**FR-9: Usage Analytics Dashboard**

**FR-9.1:** System SHALL provide users with dashboard showing:
- Total transcription minutes (daily/monthly)
- API call count
- Average processing time
- Error rate
- WER estimate (on validation set)

**FR-9.2:** System SHALL provide admin dashboard showing:
- System-wide metrics (throughput, latency)
- Model performance metrics (WER, CER)
- Infrastructure health (GPU utilization, queue depth)

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

**NFR-1: Transcription Speed**
- Real-Time Factor (RTF) < 0.3 on GPU
- RTF < 1.0 on CPU (optional deployment)
- P50 latency < 800ms for 1-min audio
- P95 latency < 1.5s for 1-min audio
- P99 latency < 2.5s for 1-min audio

**NFR-2: Throughput**
- Support 500+ concurrent transcription requests
- Process 50,000+ minutes of audio per day
- Batch processing: 10,000+ files per batch job

**NFR-3: Scalability**
- Horizontal scaling via Kubernetes
- Auto-scaling based on queue depth
- Support 10x traffic spikes (e.g., viral video)

---

### 4.2 Reliability Requirements

**NFR-4: Uptime & Availability**
- System uptime: 99.5% (43 minutes downtime/month)
- Planned maintenance windows: < 4 hours/month
- Graceful degradation: fallback to standard Whisper if fine-tuned model unavailable

**NFR-5: Error Handling**
- Automatic retry on transient failures (3 attempts)
- Clear error messages for API users
- Fallback to simpler model on OOM errors

**NFR-6: Data Durability**
- Transcription results stored with 99.99% durability (S3/GCS)
- Audio backups (if enabled) with 99.9% durability

---

### 4.3 Accuracy Requirements

**NFR-12: Word Error Rate (WER) Targets**

| Speech Type | Target WER | World-Class WER |
|-------------|------------|-----------------|
| Malaysian mixed speech | < 15% | < 12% |
| English-only (Malaysian accent) | < 8% | < 5% |
| Malay-only | < 12% | < 8% |
| Code-switching (balanced) | < 16% | < 13% |
| Noisy audio (SNR 10-15dB) | < 25% | < 20% |
| Call center audio | < 18% | < 15% |

**NFR-13: Code-Switching Accuracy**
- Language boundary detection: > 85% F1-score
- Correct language tag per word: > 80% accuracy

**NFR-14: Particle Recognition**
- Particle detection recall: > 80%
- Particle classification accuracy: > 75%

---

## 5. Technical Constraints

### 5.1 Model Constraints

**TC-1:** MUST use OpenAI Whisper-large v3 as base model (1.5B parameters)

**TC-2:** SHOULD use Unsloth for fine-tuning optimization (4x speed up)

**TC-3:** Fine-tuning dataset minimum size: 10 hours (recommended: 50+ hours)

**TC-4:** Model size limit: < 5GB for deployment efficiency

**TC-5:** Inference hardware:
- Training: NVIDIA A100/H100 40GB+ (LoRA/QLoRA for 24GB GPUs)
- Inference: NVIDIA T4 16GB or A10 24GB (or CPU with longer RTF)

---

### 5.2 Infrastructure Constraints

**TC-6:** Cloud provider flexibility: AWS, GCP, or Azure (multi-cloud ready)

**TC-7:** Kubernetes deployment for orchestration

**TC-8:** Docker containers for reproducibility

**TC-9:** Maximum audio file size: 500MB (3 hours @ 320kbps MP3)

**TC-10:** API gateway rate limiting: 1000 req/s max per instance

---

### 5.3 Data Constraints

**TC-11:** Audio format support: WAV, MP3, M4A, FLAC, OGG

**TC-12:** Sample rate: 8-48kHz (resampled to 16kHz internally)

**TC-13:** Minimum audio quality: SNR > 5dB (warn users if lower)

**TC-14:** Data retention: Default 7 days, max 30 days (PDPA compliance)

---

## 6. Assumptions & Dependencies

### 6.1 Assumptions

**AS-1:** OpenAI Whisper-large v3 license allows commercial fine-tuning and deployment (MIT license confirmed ✓)

**AS-2:** Unsloth supports Whisper architecture (confirmed ✓)

**AS-3:** Malaysian speech data is available or can be collected (10-50 hours needed)

**AS-4:** Users have stable internet for API access

**AS-5:** Majority of use cases are batch transcription (80%+), not real-time

**AS-6:** Users prefer English/Malay API documentation (localization not critical initially)

---

### 6.2 Dependencies

**DEP-1: External Libraries**
- Whisper (OpenAI)
- Unsloth (fine-tuning optimization)
- PyTorch 2.0+
- HuggingFace Transformers & Datasets
- librosa/torchaudio (audio processing)

**DEP-2: Infrastructure**
- GPU cloud provider (AWS EC2 P3/P4, GCP A100, Azure NCv3)
- S3/GCS for audio/transcript storage
- Redis for job queue management
- PostgreSQL for metadata storage

**DEP-3: Data Sources**
- Malaysian speech corpus (to be collected/licensed)
- Crowdsourced transcriptions (potential partnership)
- Existing datasets: Common Voice (if available for Malay)

**DEP-4: Third-Party Services**
- Stripe for payment processing
- SendGrid for transactional emails
- Sentry for error tracking
- DataDog/New Relic for monitoring

---

## 7. Out of Scope (Future Phases)

### Phase 2 Features (Not in MVP)

**OOS-1: Real-Time Streaming Transcription**
- WebSocket API for live audio
- Partial transcription updates
- Timeline: Month 6-9

**OOS-2: Speaker Diarization**
- "Who said what" identification
- Multi-speaker meeting transcription
- Timeline: Month 9-12

**OOS-3: Emotion/Sentiment Detection**
- Detect speaker emotion from prosody
- Sentiment labels (positive/negative/neutral)
- Timeline: Year 2

**OOS-4: Punctuation & Capitalization**
- Automatic punctuation insertion
- Proper noun capitalization
- Timeline: Month 6-9 (model fine-tuning)

**OOS-5: Custom Vocabulary**
- User-provided word lists (brand names, technical terms)
- Bias model toward custom words
- Timeline: Month 9-12

**OOS-6: Mobile SDK**
- iOS and Android native libraries
- Offline mode (on-device inference)
- Timeline: Year 2

**OOS-7: Video Processing**
- Direct video file upload (extract audio)
- YouTube URL support
- Timeline: Month 6-9

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Insufficient training data** (< 10 hours) | High | Medium | Partner with data collection services; crowdsource |
| **WER doesn't meet < 15% target** | High | Low | Iterate on data quality; try Whisper-large-v3-turbo; extend training |
| **Code-switching detection poor** | High | Low | Augment training data with synthetic code-switching |
| **Unsloth incompatibility** | Medium | Very Low | Fallback to standard HuggingFace Trainer |
| **GPU cost exceeds budget** | Medium | Medium | Use smaller GPU instances; optimize batch sizes |
| **Whisper license changes** | Low | Very Low | Monitor OpenAI announcements; have fallback plan |

---

### 8.2 Business Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Low user adoption** | High | Medium | Aggressive marketing; offer free tier; partner with content platforms |
| **Competition from big tech** (Google, AWS improve) | High | Medium | Focus on niche (Malaysian); build community; offer self-hosting |
| **Price sensitivity** (users unwilling to pay) | High | Low | Offer generous free tier; demonstrate ROI clearly |
| **Regulatory changes** (PDPA, AI laws) | Medium | Low | Design for compliance from day 1; legal consultation |
| **Data privacy concerns** | Medium | Low | Transparent privacy policy; offer immediate deletion; on-premise option |

---

### 8.3 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Infrastructure downtime** | Medium | Low | Multi-region deployment; automated failover |
| **Data breach** | High | Very Low | Encryption, security audits, penetration testing |
| **Model degradation over time** | Medium | Low | Continuous evaluation; monthly re-training |
| **Support burden** (too many user issues) | Medium | Medium | Comprehensive docs; community forum; chatbot |

---

## 9. Appendix

### 9.1 Glossary

- **ASR**: Automatic Speech Recognition (speech-to-text)
- **WER**: Word Error Rate (% of words incorrectly transcribed)
- **CER**: Character Error Rate
- **RTF**: Real-Time Factor (processing_time / audio_duration)
- **Code-switching**: Alternating between two+ languages in speech
- **Discourse particle**: Function words expressing attitude/stance (lah, leh, etc.)
- **LoRA**: Low-Rank Adaptation (efficient fine-tuning method)
- **QLoRA**: Quantized LoRA (even more memory-efficient)

### 9.2 References

1. OpenAI Whisper Paper: https://arxiv.org/abs/2212.04356
2. Malaysian English linguistics studies
3. Code-switching in Southeast Asia (academic research)
4. Unsloth documentation: https://github.com/unslothai/unsloth
5. PDPA (Malaysia): https://www.pdp.gov.my

### 9.3 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-12 | ASR Team | Initial PRD |

---

**Document Status:** Draft → Review → Approved  
**Next Review Date:** 2025-11-12  
**Owner:** Product Manager (ASR)


