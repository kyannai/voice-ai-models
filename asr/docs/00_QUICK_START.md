# Quick Start Guide
# Malaysian ASR Documentation

**Welcome!** This guide will help you navigate the comprehensive documentation suite for building a Malaysian Automatic Speech Recognition (ASR) system that understands Malay with English code-switching, Pinyin, slang, and particles.

---

## 📚 Document Overview

I've created **9 detailed documents**, each focusing on a specific aspect of fine-tuning Whisper-large v3 for Malaysian speech:

### 1. [Product Requirements Document (PRD)](01_PRD_Product_Requirements.md)
- Product vision and market analysis
- User personas and use cases for Malaysian ASR
- Functional and non-functional requirements
- Success metrics and KPIs (WER, CER, code-switching accuracy)
- Competitive analysis (ElevenLabs, Google, AWS, Azure ASR)

### 2. [Data Preparation Guide](03_Data_Preparation_Guide.md)
- Data collection strategies for Malaysian speech
- Audio recording specifications (16kHz, mono, quality standards)
- Transcription guidelines (handling code-switching, particles)
- Malaysian-specific annotation schema
- Quality control procedures
- Data augmentation techniques

### 3. [Training Strategy & Guide](04_Training_Strategy_Guide.md)
- Whisper-large v3 architecture and Unsloth integration
- Fine-tuning pipeline (LoRA, QLoRA strategies)
- Audio preprocessing and feature extraction
- Hyperparameter tuning for Malaysian speech
- Multi-stage training approach
- Training optimization and debugging

### 4. [Evaluation Methodology](05_Evaluation_Methodology.md)
- Objective metrics (WER, CER, F1-score)
- Malaysian-specific evaluation (code-switching accuracy, particle recognition)
- Subjective evaluation protocols
- Test set design and stratification
- Continuous evaluation in production
- Benchmark comparisons

### 5. [Deployment Guide](06_Deployment_Guide.md)
- Model inference architecture and optimization
- API design and implementation
- Infrastructure setup (AWS/GCP/Azure GPU instances)
- Docker containerization with Whisper
- Kubernetes deployment for scale
- WebSocket API for streaming transcription
- CI/CD pipeline (GitHub Actions)
- Monitoring (Prometheus, Grafana, WER tracking)

### 6. [Project Execution Plan & Timeline](07_Project_Execution_Plan.md)
- Detailed 8-week timeline for production deployment
- Phase-by-phase breakdown (data → training → deployment)
- Resource allocation and budget ($50K-$65K balanced option)
- Risk management for multilingual ASR
- Decision gates and success criteria
- Team structure and roles

### 7. [Shared Data Strategy](08_Shared_Data_Strategy.md)
- Cost-saving strategy: one recording session for both ASR and TTS
- Master recording specifications (48kHz/24-bit)
- Processing pipeline for ASR (16kHz) and TTS (22.05kHz)
- Cost savings breakdown ($4,500-8,000 saved!)
- Implementation guide with code examples

### 8. [Data Collection Quickstart](09_Data_Collection_Quickstart.md)
- Week-by-week data collection checklist
- Recording session preparation guide
- Tools and commands reference
- Quality control procedures
- Troubleshooting common issues

### 9. [Scalability Notes](10_Scalability_Notes.md)
- Production-ready architecture for scalable deployment
- Auto-scaling strategies and load balancing
- Performance optimization techniques
- Cost management for cloud infrastructure
- Monitoring and alerting setup

---

## 🎯 Quick Start: Zero to Production

### Option A: "I Just Want to Fine-tune ASAP" (Fast Track)
**Time needed:** 1-2 days

```bash
# 1. Environment Setup (30 min)
conda create -n whisper-malay python=3.10
conda activate whisper-malay
pip install unsloth torch torchaudio datasets

# 2. Prepare Data (4-8 hours with existing audio)
# See: 03_Data_Preparation_Guide.md Section 11

# 3. Fine-tune with Unsloth (4-12 hours depending on dataset)
# See: 04_Training_Strategy_Guide.md Section 4

# 4. Evaluate (1-2 hours)
# See: 05_Evaluation_Methodology.md Section 2

# 5. Deploy (2-4 hours)
# See: 06_Deployment_Guide.md Section 3
```

**Documents to read:**
- [Training Strategy - Quick Start](04_Training_Strategy_Guide.md#1-overview) (30 min)
- [Data Preparation - Quick Start](03_Data_Preparation_Guide.md#11-quick-start-guide) (20 min)
- [Deployment - Docker Setup](06_Deployment_Guide.md#3-docker-containerization) (20 min)

---

### Option B: "I Want to Understand Everything" (Complete Track)
**Time needed:** 1-2 weeks

**Week 1: Research & Preparation**
- Day 1-2: Read all documentation (12-15 hours)
- Day 3-4: Collect and prepare Malaysian speech data
- Day 5: Set up training infrastructure

**Week 2: Training & Deployment**
- Day 1-3: Fine-tune Whisper with Unsloth
- Day 4: Evaluate and iterate
- Day 5: Deploy to production

**Documents to read:** All 7 documents in order

---

## 🌟 What Makes This ASR Special?

### 1. **Primary Malay with Code-Switching Support** 🇲🇾
```
Input Audio: "Saya nak tolong check system ni lah, I think ada problem sikit"
Output: "Saya nak tolong check system ni lah, I think ada problem sikit"

Word-by-word breakdown:
Saya    nak     tolong  check   system  ni      lah     I       think   ada     problem sikit
[Malay] [Malay] [Malay] [Eng]   [Eng]   [Malay] [Part.] [Eng]   [Eng]   [Malay] [Eng]   [Malay]
```

Handles the natural Malaysian speech pattern:
- **Primary**: Malay grammar and vocabulary (60-70% of speech)
- **Secondary**: English words mixed in (20-30%)
- **Additional**: Pinyin/Mandarin terms, local slang, special vocabulary
- **Particles**: Discourse markers throughout (lah, leh, loh, meh, lor)

### 2. **Particle-Aware Recognition** 
Proper handling of Malaysian discourse particles:
- **lah** (assertion, familiarity) - "Okay lah"
- **leh** (suggestion, possibility) - "Can try leh"
- **loh** (obviousness, reminder) - "I told you loh"
- **meh** (doubt, surprise) - "Really meh?"
- **lor** (resignation, acceptance) - "Cannot help lor"

### 3. **Malaysian Accent Understanding**
- Trained specifically on Malaysian Malay pronunciation patterns
- Recognizes Malaysian English accent when code-switching
- Handles local accent variations across different regions (KL, Penang, Johor, etc.)
- Understands prosody and intonation unique to Malaysian speech

---

## 📊 Expected Performance Targets

| Metric | Target | World-Class |
|--------|--------|-------------|
| **Overall WER** | <15% | <10% |
| **Code-Switch Accuracy** | >85% | >90% |
| **Particle Recognition** | >80% | >85% |
| **English-only WER** | <8% | <5% |
| **Malay-only WER** | <12% | <8% |
| **Real-Time Factor (RTF)** | <0.3 | <0.2 |
| **API Latency (p95)** | <1.5s | <1.0s |

---

## 🛠️ Technology Stack

### Core ML Stack
- **Base Model:** OpenAI Whisper-large v3 (1.5B parameters)
- **Fine-tuning:** Unsloth (LoRA/QLoRA optimization)
- **Framework:** PyTorch 2.0+, HuggingFace Transformers
- **Data Processing:** librosa, torchaudio, datasets

### Infrastructure
- **Training:** NVIDIA A100/H100 (40GB+ VRAM)
- **Inference:** NVIDIA T4/A10 or CPU with optimizations
- **Orchestration:** Kubernetes, Docker
- **Monitoring:** Prometheus, Grafana, Weights & Biases

### Development
- **Language:** Python 3.10+
- **API Framework:** FastAPI, WebSocket
- **Testing:** pytest, Hypothesis
- **CI/CD:** GitHub Actions, Docker Hub

---

## 🎓 Learning Resources

### Before You Start
**Essential Background:**
1. **Whisper Architecture** (30 min)
   - [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
   - [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

2. **LoRA & Fine-tuning** (20 min)
   - [LoRA Paper](https://arxiv.org/abs/2106.09685)
   - [Unsloth Documentation](https://github.com/unslothai/unsloth)

3. **Malaysian Linguistics** (15 min)
   - Code-switching patterns in Malaysian English
   - Function of discourse particles

### During Development
- Whisper troubleshooting guide (in Training Strategy doc)
- Malaysian speech corpus examples
- Evaluation best practices

---

## 📞 Support & Questions

### Common Questions

**Q: Can I use a smaller Whisper model?**  
A: Yes! You can use Whisper-medium or Whisper-small for faster training/inference. See [Technical Architecture - Model Selection](02_Technical_Architecture.md#2-model-architecture).

**Q: How much data do I need?**  
A: Minimum 10 hours, recommended 50+ hours of transcribed Malaysian speech. See [Data Preparation - Dataset Size](03_Data_Preparation_Guide.md#3-dataset-requirements).

**Q: Does Unsloth really work with Whisper?**  
A: Yes! Unsloth fully supports Whisper models. See [Training Strategy - Unsloth Setup](04_Training_Strategy_Guide.md#3-unsloth-setup).

**Q: What GPU do I need?**  
A: For training: A100 40GB (recommended) or RTX 4090 24GB. For inference: T4 16GB or CPU. See [Technical Architecture - Hardware](02_Technical_Architecture.md#5-infrastructure).

**Q: How long does training take?**  
A: With Unsloth: 6-12 hours for 50 hours of data on A100. Without: 24-48 hours. See [Training Strategy - Timeline](04_Training_Strategy_Guide.md#8-training-timeline).

---

## 🚦 Getting Started Checklist

### Prerequisites
- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ (for GPU training)
- [ ] Access to GPU (local or cloud)
- [ ] 10+ hours of Malaysian speech data (or plan to collect)

### Setup Steps
- [ ] Read Quick Start (this document) ✓
- [ ] Read PRD to understand requirements
- [ ] Read Technical Architecture for system overview
- [ ] Set up development environment
- [ ] Prepare/collect Malaysian speech data
- [ ] Follow Training Strategy guide
- [ ] Evaluate model performance
- [ ] Deploy to production

### First Milestone Goals
- [ ] Fine-tune Whisper on 10+ hours of data
- [ ] Achieve <20% WER on test set
- [ ] Successfully transcribe code-switched speech
- [ ] Deploy basic API endpoint
- [ ] Run first user acceptance test

---

## 📈 Success Milestones

### Phase 1: Proof of Concept (Weeks 1-4)
- ✅ Environment set up
- ✅ 10 hours of data collected and prepared
- ✅ First model trained with Unsloth
- ✅ WER < 25% achieved
- ✅ Code-switching demo working

### Phase 2: Alpha (Weeks 5-8)
- ✅ 30+ hours of data
- ✅ WER < 18% achieved
- ✅ API deployed locally
- ✅ 5 alpha testers onboarded

### Phase 3: Beta (Weeks 9-16)
- ✅ 50+ hours of data
- ✅ WER < 15% achieved
- ✅ Production deployment
- ✅ 50+ beta testers

### Phase 4: Production Launch (Weeks 17-24)
- ✅ 100+ hours of data
- ✅ WER < 12% achieved
- ✅ Public API launch
- ✅ 99.9% uptime SLA

---

## 🎯 Next Steps

### Immediate Actions (Today)
1. **Read this Quick Start** ✓
2. **Skim the [PRD](01_PRD_Product_Requirements.md)** to understand what you're building
3. **Read [Technical Architecture](02_Technical_Architecture.md)** for system overview
4. **Bookmark key sections** for your role

### This Week
1. **Set up environment** following [Training Strategy - Setup](04_Training_Strategy_Guide.md#2-environment-setup)
2. **Collect/prepare 10 hours** of Malaysian speech data using [Data Preparation Guide](03_Data_Preparation_Guide.md)
3. **Run first training experiment** with Unsloth

### This Month
1. **Complete Phase 1** (PoC) milestones
2. **Iterate on data quality** and model performance
3. **Set up evaluation pipeline**
4. **Plan Phase 2** (Alpha) goals

---

## 📚 Document Reading Order

### For First-Time Readers
1. **Start here** → 00_QUICK_START.md (this doc) ✓
2. → [01_PRD_Product_Requirements.md](01_PRD_Product_Requirements.md)
3. → [02_Technical_Architecture.md](02_Technical_Architecture.md)
4. → [03_Data_Preparation_Guide.md](03_Data_Preparation_Guide.md)
5. → [04_Training_Strategy_Guide.md](04_Training_Strategy_Guide.md)
6. → [05_Evaluation_Methodology.md](05_Evaluation_Methodology.md)
7. → [06_Deployment_Guide.md](06_Deployment_Guide.md)
8. → [07_Project_Execution_Plan.md](07_Project_Execution_Plan.md)

### For Returning Readers
Jump directly to the section you need using the navigation above.

---

**Let's build world-class Malaysian ASR together! 🚀🇲🇾**

*Last updated: October 12, 2025*

