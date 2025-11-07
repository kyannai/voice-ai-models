# Project Execution Plan & Timeline
# Malaysian Multilingual ASR System

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Timeline](#2-project-timeline)
3. [Phase Breakdown](#3-phase-breakdown)
4. [Resource Allocation](#4-resource-allocation)
5. [Budget Estimates](#5-budget-estimates)
6. [Risk Management](#6-risk-management)
7. [Decision Gates](#7-decision-gates)
8. [Success Criteria](#8-success-criteria)

---

## 1. Executive Summary

### 1.1 Project Overview

**Goal:** Build and deploy a production-grade Malaysian multilingual ASR system using Whisper-large v3, fine-tuned with Unsloth, achieving <15% WER on Malaysian code-switching speech.

**Timeline:** 2 months (8 weeks, balanced speed & quality)  
**Team Size:** 3-4 people (focused team)  
**Budget:** $40K-$60K (balanced investment)  
**Approach:** Mix existing datasets + targeted Malaysian data collection, Unsloth fine-tuning

**Key Objectives:**
- ⏱️ **Reasonable timeline**: 8 weeks allows proper development & testing
- 🎯 **Production-ready**: Scalable architecture with thorough QA
- 💰 **Balanced budget**: Leverage existing data + collect 30-40 hours Malaysian speech
- 📊 **Quality target**: <15% WER (achievable with 2 months)
- ✅ **Proper validation**: Time for beta testing and iteration before launch

### 1.2 Key Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| **M1: Foundation & Data** | End of Week 2 | Environment ready, 30-40hrs Malaysian data collected |
| **M2: Model Trained** | End of Week 5 | Fine-tuned model, WER < 15%, thorough evaluation |
| **M3: Beta Testing** | End of Week 6 | 10-20 beta users, feedback collected |
| **M4: Production Ready** | End of Week 7 | Production API deployed, load tested, documented |
| **M5: Public Launch** | End of Week 8 | Scalable deployment, monitoring, public access |

### 1.3 Success Metrics

**Technical Metrics (8-Week Target):**
- Word Error Rate (WER): < 15% (production-grade)
- Code-switching F1: > 85%
- Particle recall: > 80%
- Real-Time Factor: < 0.3
- API uptime: > 99.5% (high reliability)
- System can scale to 50-100 concurrent requests (voice is secondary feature)

**Business Metrics (Post-Launch):**
- Beta users: 20-50 (quality early adopters)
- Target: 500-2,000 voice users initially (5-10% of text ChatGPT users)
- System designed to scale to 10,000+ users without re-architecture
- Infrastructure cost: < $500/month initially (voice is secondary feature)
- Ready for integration with main text product

---

## 2. Project Timeline

### 2.1 Gantt Chart (8 Weeks - Balanced)

```
Weeks 1-2: Foundation & Data Collection
│████████████████████████████████│ Setup, Data Collection, Baseline
                                 
Weeks 3-5: Model Training & Optimization
        │████████████████████████████████████████│ Fine-tuning, Evaluation
                                                 
Week 6: Beta Testing & Iteration
                        │████████████████│ Beta Users, Feedback
                                         
Week 7: Production Preparation
                                │████████████████│ API, Testing
                                                 
Week 8: Deployment & Launch
                                        │████████████████│ Deploy

Timeline: Week 1────2────3────4────5────6────7────8 (Launch)
```

### 2.2 Detailed 8-Week Timeline

**Weeks 1-2: Foundation & Data Collection**
- **Week 1, Days 1-3**: Team kickoff, GPU setup (A100), Unsloth installation
- **Week 1, Days 4-7**: Baseline Whisper testing, Common Voice Malay download
- **Week 2, Days 1-3**: Recruit 5-10 voice actors for Malaysian speech
- **Week 2, Days 4-7**: Record 20-30 hours of targeted Malaysian data
- **Milestone M1**: 30-40 hours total data ready (Common Voice + custom recordings)

**Weeks 3-5: Model Training & Optimization**
- **Week 3**: Data preprocessing, create HuggingFace datasets, start LoRA training
- **Week 4**: Continue training (3-5 epochs), monitor WER on validation set
- **Week 5**: Complete training, full evaluation, hyperparameter tuning
- **Milestone M2**: Model achieves WER < 15%, code-switching F1 > 85%

**Week 6: Beta Testing & Iteration**
- **Days 1-2**: Deploy to staging environment, create simple API
- **Days 3-5**: Onboard 10-20 beta users, collect feedback
- **Days 6-7**: Iterate on model or API based on feedback
- **Milestone M3**: Beta testing complete, critical issues fixed

**Week 7: Production Preparation**
- **Days 1-3**: Production API development (FastAPI + Celery)
- **Days 4-5**: Docker containerization, Kubernetes setup
- **Days 6-7**: Load testing (50-100 concurrent for voice feature), security hardening
- **Milestone M4**: Production infrastructure ready, tested

**Week 8: Deployment & Launch**
- **Days 1-2**: Production deployment, monitoring setup (Prometheus/Grafana)
- **Days 3-4**: Smoke tests, documentation finalization
- **Day 5**: Soft launch to beta users for final validation
- **Days 6-7**: Public launch, marketing, initial user onboarding
- **Milestone M5**: Production system live, monitored, publicly accessible

---

## 3. Phase Breakdown (8-Week Balanced Timeline)

### 3.1 Weeks 1-2: Foundation & Data Collection (Shared with TTS!)

**Objectives:**
- Set up production infrastructure
- **Plan data collection strategy** (new recordings OR existing data)
- **Generate sentences and prepare recording infrastructure**
- Collect 30-40 hours Malaysian speech at **48kHz/24-bit** (reusable for TTS!)
- Establish baseline and validate approach

**Critical Tasks:**

| Week | Day | Task | Owner | Output |
|------|-----|------|-------|--------|
| **Week 1** | 1-2 | **Data Strategy Decision** | ML Lead + Data Lead | Strategy document |
| | | • Assess existing datasets (Common Voice, etc.) | Data Lead | Dataset inventory |
| | | • Decide: New recordings vs Existing data vs Hybrid | ML Lead | Go/No-Go decision |
| | | • Define requirements (30-40hrs, 48kHz/24-bit for both ASR+TTS) | Data Lead | Requirements doc |
| **Week 1** | 3-4 | **Sentence Generation & Planning** | Data Lead | 10K-12K sentences |
| | | • Generate sentences with GPT-4 (Malaysian patterns) | Data Lead | 8,000 sentences |
| | | • Curate from Common Voice / Malaysian corpus | Data Lead | 2,000 sentences |
| | | • Validate sentence quality & diversity | Data Lead | QA passed |
| | | • Assign sentences to speakers (unique per speaker) | Data Lead | Assignment CSV |
| **Week 1** | 5-7 | **Recording Infrastructure Setup** | ML Engineer + Data Lead | Ready to record |
| | | • **Option A: Professional Studio (Recommended)** | | |
| | | ├─ Prepare recording scripts (CSV format) | Data Lead | Scripts ready |
| | | └─ Recruit 3-5 voice actors ($200/hr bulk rate) | Data Lead | Actors contracted |
| | | • **Option B: Recording UI (if remote actors)** | | |
| | | ├─ Build simple recording web interface | ML Engineer | UI deployed |
| | | └─ Test with sample recordings | Data Lead | UI validated |
| | | • **GPU Environment Setup (Parallel)** | ML Engineer | Environment ready |
| | | ├─ Provision A100 GPU, install Unsloth | ML Engineer | GPU operational |
| | | ├─ Test baseline Whisper-large v3 | ML Engineer | Baseline WER measured |
| | | └─ Download Common Voice Malay (20hrs) | Data Lead | Free dataset ready |
| **Week 2** | 1-5 | **Data Collection Execution** | Audio Eng + Data Lead | 30-40hrs at 48kHz |
| | | • **Record at 48kHz/24-bit WAV** (for both ASR+TTS!) | Audio Engineer | Master recordings |
| | | • Studio sessions: 3-5 speakers × 10-13hrs each | Audio Engineer | High quality audio |
| | | • Real-time QA during recording sessions | Audio Engineer | Quality checked |
| | | • Immediate backup to cloud storage | Data Lead | Backed up |
| **Week 2** | 6-7 | **Data Processing & Preparation** | ML Engineer | Datasets ready |
| | | • Process master 48kHz → 16kHz for ASR | ML Engineer | ASR dataset (16kHz) |
| | | • Keep master 48kHz for TTS (process later to 22.05kHz) | ML Engineer | TTS master saved |
| | | • Create metadata for ASR training | ML Engineer | metadata.csv |
| | | • Train/val/test splits (80/10/10) | ML Engineer | Splits ready |
| | | • Final validation: 30-40 hours total | Data Lead | ✅ Milestone M1 |

**Deliverables:**
- ✅ **Data Strategy Document** (recording plan, budget, timeline)
- ✅ **Sentence Pool:** 10,000-12,000 Malaysian sentences generated & validated
- ✅ **Recording Infrastructure:** Studio scripts OR recording UI deployed
- ✅ **GPU environment operational** (A100 + Unsloth)
- ✅ **Baseline Whisper-large v3:** WER measured on Malaysian test data
- ✅ **30-40 hours MASTER recordings** at 48kHz/24-bit:
  - 20hrs from Common Voice (free, already available)
  - 10-20hrs custom recordings (professional voice actors)
  - **Master recordings saved for TTS reuse (Month 3-4)!**
- ✅ **ASR-ready dataset:** 30-40 hours at 16kHz (processed from master)
- ✅ **Train/val/test splits created** (80/10/10)

**Budget (Weeks 1-2):** $10,000-$14,000
- Voice actors: $3,000-4,000 (10-20hrs × $200/hr bulk rate)
- Recording UI dev (optional): $1,000-2,000 (if remote recording)
- OR Studio rental: $1,000 (if traditional recording)
- GPU (A100): $2,500 (100 hours × $25/hr)
- Salaries (3 people × 2 weeks): $9,000
- Tools & licenses: $500

**Key Decisions:**

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Recording Strategy** | New studio recordings | Use existing data only | **Hybrid (20hrs CV + 10-20hrs custom)** ✓ |
| **Recording Quality** | 48kHz/24-bit (future-proof) | 16kHz/16-bit (ASR-only) | **48kHz/24-bit for TTS reuse** ✓ |
| **Recording Method** | Professional studio | Remote UI recording | **Studio (better quality)** ✓ |
| **Sentence Assignment** | Different per speaker | Same for all speakers | **Different (maximize diversity)** ✓ |

**⚡ KEY OPTIMIZATION:** Recording at 48kHz/24-bit enables reuse for TTS training in Months 3-4, saving $6,000-10,000!

---

### 3.2 Weeks 3-5: Model Training & Optimization

**Objectives:**
- Fine-tune Whisper-large v3 with Unsloth (LoRA)
- Achieve production-grade WER < 15%
- Thorough evaluation and hyperparameter tuning

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **8-9** | Start LoRA fine-tuning (r=64) | ML Engineer | Training running |
| **10-11** | Monitor loss, adjust hyperparameters | ML Engineer | Stable training curve |
| **12** | Complete training (3 epochs) | ML Engineer | Final checkpoint |
| **13** | Evaluate: WER, code-switching, particles | ML Engineer | WER < 18% confirmed |
| **14** | Model optimization (quantization, export) | ML Engineer | Production-ready model |

**Deliverables:**
- ✅ Fine-tuned Whisper-large v3 with Malaysian LoRA adapters
- ✅ WER < 18% on test set (10-15% stretch goal)
- ✅ Code-switching F1 > 80%
- ✅ Particle recall > 75%
- ✅ Model exported in multiple formats (PyTorch, ONNX optional)

**Budget (Week 2):** $3,500-$5,000
- GPU (A100): $2,500 (100 hours × $25/hr for continuous training)
- Salaries (3 people × 1 week): $4,500

**Training Configuration:**
```python
# Unsloth LoRA config for 4-day training
r=64, lora_alpha=128
batch_size=4, gradient_accumulation=8
learning_rate=5e-5, warmup_steps=500
total_steps=~10,000 (for 20 hours data)
```

**Key Decisions:**
- Train on 15-20 hours initially (faster iteration)
- 3 epochs vs 5? → **3 epochs** (sufficient with good data)
- QLoRA vs LoRA? → **LoRA** (better quality, A100 has enough VRAM)

---

### 3.3 Week 3: API Development & Integration (Days 15-21)

**Objectives:**
- Build production-grade FastAPI
- Containerize with Docker
- Test at scale (100+ concurrent)

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **15-16** | FastAPI + Celery worker setup | Backend Dev | Sync + async endpoints |
| **17** | Docker containerization | DevOps | Images built and tagged |
| **18** | Integration testing with model | Backend Dev | End-to-end flow working |
| **19-20** | Load testing (100 concurrent requests) | DevOps | Performance benchmarked |
| **21** | Security hardening + API docs (Swagger) | Backend Dev | Production-ready API |

**Deliverables:**
- ✅ FastAPI with `/v1/transcribe` (sync) and `/v1/transcribe/async` endpoints
- ✅ Celery workers for background processing
- ✅ Docker images (API + Worker + Redis)
- ✅ Load tested: 100+ concurrent, RTF < 0.3
- ✅ API documentation (Swagger UI)
- ✅ Authentication (API key system)

**Budget (Week 3):** $2,500-$4,000
- Infrastructure (staging T4 GPU): $500
- Salaries (3 people × 1 week): $4,500

**API Stack:**
```python
# Key components
FastAPI + Uvicorn (API server)
Celery + Redis (async job queue)
PostgreSQL (user metadata)
S3/GCS (audio storage)
Prometheus (metrics)
```

**Key Decisions:**
- Synchronous vs async-only? → **Both** (sync for <1min audio, async for longer)
- WebSocket streaming? → **Not for MVP** (add in v1.1)
- Authentication: API keys (simple, fast to implement)

---

### 3.4 Week 4: Production Deployment & Launch (Days 22-28)

**Objectives:**
- Deploy to Kubernetes (production)
- Set up monitoring and alerting
- Launch publicly with initial users

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **22-23** | Kubernetes deployment (prod) | DevOps | K8s cluster configured |
| **24** | Monitoring: Prometheus + Grafana | DevOps | Dashboards live |
| **25** | Security review + SSL setup | DevOps | HTTPS enabled, secured |
| **26** | Production smoke tests | All | System validated |
| **27** | Documentation + onboarding flow | Backend Dev | User-ready docs |
| **28** | Public launch + initial user testing | All | 5-10 users onboarded |

**Deliverables:**
- ✅ Production Kubernetes cluster (GKE/EKS)
- ✅ Auto-scaling configured (2-10 worker pods)
- ✅ Monitoring dashboards (latency, WER, errors)
- ✅ Alert rules configured (Slack/email)
- ✅ Production domain: `api.asr.example.com`
- ✅ Public documentation site
- ✅ 5-10 initial users successfully using system

**Budget (Week 4):** $3,000-$5,000
- Infrastructure (production): $1,500 (T4 GPU + API servers)
- Salaries (3 people × 1 week): $4,500
- Domain + SSL: $50

**Production Stack:**
```
Kubernetes (GKE/EKS)
├── API pods (3 replicas)
├── Worker pods (2-10, auto-scale)
├── Redis (job queue)
└── PostgreSQL (metadata)

Monitoring:
- Prometheus (metrics)
- Grafana (dashboards)
- Sentry (error tracking)
```

**Launch Checklist:**
- [ ] WER < 18% validated
- [ ] API load tested (100 concurrent)
- [ ] Monitoring operational
- [ ] Documentation complete
- [ ] 5 test users successfully transcribed audio
- [ ] Pricing model defined (even if free initially)

**Post-Launch (Weeks 5-8):**
- Continuous monitoring and optimization
- Collect user feedback
- Iterate on model (re-train if needed)
- Scale infrastructure as users grow
- Plan v1.1 features (streaming, diarization)

---

## 4. Resource Allocation (4-Week Timeline)

### 4.1 Lean Team Structure

**Core Team (3-4 people, full-time):**

| Role | Responsibilities | FTE | Weeks |
|------|------------------|-----|-------|
| **ML Engineer (Lead)** | Training, optimization, evaluation | 1.0 | 4 weeks |
| **Backend/DevOps** | API, Docker, K8s deployment | 1.0 | 4 weeks |
| **Full-Stack Engineer** | API, frontend docs, integration | 0.75 | 4 weeks |
| **Project Lead** (optional) | Coordination, decisions | 0.25 | 4 weeks |

**Total FTE:** ~3 people working full-time for 4 weeks

**Why This Works:**
- ⚡ **Focused scope**: No custom data collection, use existing datasets
- 🎯 **Production-grade, not perfect**: 18% WER acceptable for v1.0
- 💪 **Multi-skilled team**: Backend engineer handles DevOps
- 🚀 **Rapid iteration**: Daily standups, no bureaucracy

**NO External Resources Needed:**
- ❌ No voice actors (use Common Voice dataset)
- ❌ No transcribers (data already transcribed)
- ❌ No QA team (automated testing + ML engineer validates)
- ❌ No separate PM (lead engineer makes decisions)

### 4.2 Skill Requirements

**Must-Have (Critical):**
- **ML Engineer**: PyTorch, HuggingFace, Whisper, Unsloth/LoRA, WER evaluation
- **Backend/DevOps**: Python, FastAPI, Docker, Kubernetes (GKE/EKS), Prometheus
- **Full-Stack**: JavaScript/React (for docs site), API integration, testing

**Nice-to-Have (Not Critical for 4 weeks):**
- Linguistics background → Use online resources for particles
- Production ML experience → Documentation covers best practices
- Kubernetes expert → Follow deployment guides

---

## 5. Budget Estimates (8-Week Timeline)

### 5.1 Budget Breakdown

**Personnel Costs (8 Weeks):**

| Role | Weekly Rate | Weeks | Total |
|------|-------------|-------|-------|
| ML Engineer (Lead) | $3,000 | 8 | $24,000 |
| Backend/DevOps Engineer | $2,800 | 8 | $22,400 |
| Full-Stack Engineer (0.75 FTE) | $2,500 | 8 | $15,000 |
| Project Lead (0.25 FTE, optional) | $1,500 | 8 | $3,000 |
| **Subtotal** | | | **$64,400** |

**Infrastructure (8 Weeks):**

| Item | Cost | Notes |
|------|------|-------|
| Training GPU (A100, 300 hours) | $7,500 | Weeks 3-5: $25/hr × 300 hrs |
| Production GPU (T4, 4 weeks) | $2,000 | Weeks 5-8: $0.50/hr × 4032 hrs (continuous) |
| Kubernetes cluster (GKE/EKS) | $800 | Control plane + nodes, 8 weeks |
| Database (PostgreSQL) | $160 | Managed service, 8 weeks |
| Storage (S3/GCS) | $100 | 1TB for audio files |
| Domain + SSL | $50 | Domain + cert |
| **Subtotal** | | **$10,610** |

**Data Collection:**
| Item | Cost | Notes |
|------|------|-------|
| Common Voice Malay dataset | $0 | Free, open-source (20hrs) |
| Voice actors (20-30hrs recording) | $3,000 | $150/hr professional Malaysian speakers |
| Transcription QA | $500 | Quality checks |
| **Subtotal** | | **$3,500** |

**Tools & Services:**
| Item | Cost | Notes |
|------|------|-------|
| Monitoring (Prometheus/Grafana) | $0 | Self-hosted |
| GitHub Actions | $0 | Free tier |
| Development tools | $200 | IDEs, testing tools |
| **Subtotal** | | **$200** |

**Contingency (10% for unknowns):** $7,871

**TOTAL 8-WEEK BUDGET:** **$86,581**

### 5.2 Budget Options for 8-Week Timeline

**Option 1: Lean MVP ($35K-$45K)** 
- 2-3 people (ML Engineer + Full-Stack, part-time DevOps)
- RTX 4090 or reduced A100 hours
- Self-host or use free-tier Kubernetes
- 20 hours data only (Common Voice + 5hrs custom)
- Target: WER < 18%, basic API

**Option 2: Balanced Production ($50K-$65K)** ← **Recommended**
- 3 people full-time (ML + Backend/DevOps + Full-Stack)
- A100 for training, T4 for inference
- Managed Kubernetes (GKE/EKS)
- 30-40 hours data (Common Voice + 20hrs custom)
- Full monitoring, beta testing phase
- Target: WER < 15%, scalable API

**Option 3: Premium Quality ($75K-$95K)**
- 4 people including project lead
- Best hardware (A100 training, A10 inference)
- Enterprise-grade infrastructure
- 50+ hours high-quality data
- Extensive testing and documentation
- Target: WER < 13%, enterprise-ready

**Comparison:**

| Item | Lean | Balanced | Premium |
|------|------|----------|---------|
| **Budget** | $35-45K | $50-65K | $75-95K |
| **Team** | 2-3 people | 3 people | 4 people |
| **Timeline** | 10 weeks | 8 weeks | 8 weeks |
| **WER Target** | < 18% | < 15% | < 13% |
| **Data Hours** | 20 hrs | 30-40 hrs | 50+ hrs |
| **Infrastructure** | Basic | Production | Enterprise |
| **Risk** | Medium | Low | Very Low |

---

## 6. Risk Management (8-Week Timeline)

### 6.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **WER doesn't reach <15%** | Low-Medium | High | More data (increase to 50hrs), longer training, use Whisper-medium as fallback |
| **Voice actor delays** | Medium | Medium | Start recruiting Week 1, have backup actors, reduce custom data to 15hrs |
| **Training takes longer** | Low | Medium | Extended timeline accommodates, have backup GPU provider |
| **Team member unavailable** | Low | Medium | 8-week buffer allows for cross-training, documentation |
| **Infrastructure issues (K8s, GPU)** | Low | Medium | Test infrastructure Weeks 1-2, have multi-cloud setup |
| **Scope creep** | Low | Low | 8 weeks allows some flexibility, but strict MVP focus |
| **Data quality issues** | Medium | Medium | Rigorous QA process, 90% accuracy threshold |
| **Security vulnerabilities** | Low | High | Time for security audit Week 7, penetration testing |

### 6.2 Mitigation Strategies (Balanced Approach)

**For WER Target:**
- **Primary**: Whisper-large v3 with 30-40 hours quality data
- **Backup Plan**: Collect additional 10-20 hours if needed (budget allows)
- **Target**: <15% WER achievable with proper training time

**For Data Collection:**
- **Week 1**: Start voice actor recruitment immediately
- **Week 2**: Record 20-30 hours with professional setup
- **Quality control**: 90%+ transcription accuracy before training

**For Timeline:**
- **Weekly milestones**: Clear checkpoints every week
- **Parallel workstreams**: Data collection || Environment setup (Week 1-2)
- **Buffer time**: 8 weeks allows for 1-2 week contingency

**For Team Capacity:**
- **Cross-training**: Everyone understands all components
- **Documentation**: Comprehensive docs from day 1
- **Knowledge sharing**: Weekly demo sessions

---

## 7. Decision Gates (8-Week Timeline)

### 7.1 Go/No-Go Criteria

**Gate 1: End of Week 2**
- ✅ GPU environment operational (A100 + Unsloth installed)
- ✅ Baseline Whisper WER measured on Malaysian test set
- ✅ 30-40 hours Malaysian data collected and validated
- ✅ Train/val/test splits created
- **Decision:** Proceed to training OR collect more data

**Gate 2: End of Week 4 (Mid-Training)**
- ✅ Training progressing (epoch 2-3 complete)
- ✅ Loss decreasing steadily
- ✅ Preliminary WER < 20% on validation set
- ✅ No major technical issues
- **Decision:** Continue training OR adjust hyperparameters

**Gate 3: End of Week 5 (Model Complete)**
- ✅ WER < 15% on test set
- ✅ Code-switching F1 > 85%
- ✅ Particle recall > 80%
- ✅ Model exported and optimized
- **Decision:** Proceed to beta testing OR extend training 1 week

**Gate 4: End of Week 6 (Beta Testing)**
- ✅ Beta users onboarded (10-20 people)
- ✅ Feedback collected and analyzed
- ✅ Critical issues identified and prioritized
- ✅ Staging environment stable
- **Decision:** Proceed to production OR iterate on model

**Gate 5: End of Week 7 (Production Ready)**
- ✅ Production API functional (FastAPI + Celery)
- ✅ Load tested (50-100 concurrent requests for voice feature)
- ✅ Docker + Kubernetes deployment working
- ✅ Monitoring and alerting operational
- **Decision:** Launch Week 8 OR delay 1 week for critical fixes

**Gate 6: Week 8 Launch Decision**
- ✅ All production systems tested
- ✅ Documentation complete
- ✅ Beta users report satisfaction
- ✅ Security audit passed
- **Decision:** Public launch OR soft launch with limited users

**Contingency Plans:**
- If WER > 18% at Week 5 → Collect 10-20 more hours, extend training 1 week
- If beta feedback critical → Extend beta phase 1 week
- If infrastructure issues → Have backup cloud provider ready

---

## 8. Success Criteria (8-Week Launch)

### 8.1 Week-by-Week Success Metrics

**Week 1: Environment Setup**
- [ ] GPU environment ready (A100 provisioned)
- [ ] Unsloth + dependencies installed
- [ ] Baseline Whisper tested on Malaysian data
- [ ] Voice actor recruitment started

**Week 2: Data Collection**
- [ ] 30-40 hours Malaysian data collected
- [ ] Common Voice Malay downloaded (20hrs)
- [ ] Custom recordings complete (20-30hrs)
- [ ] Data QA: 90%+ accuracy on transcriptions

**Weeks 3-4: Training (Part 1)**
- [ ] Training started (LoRA on Whisper-large v3)
- [ ] Epoch 1-2 complete
- [ ] Validation WER < 20%
- [ ] No major technical issues

**Week 5: Training Complete**
- [ ] Training finished (3-5 epochs)
- [ ] **WER < 15%** on test set
- [ ] Code-switching F1 > 85%
- [ ] Particle recall > 80%
- [ ] Model exported and optimized

**Week 6: Beta Testing**
- [ ] Staging environment deployed
- [ ] 10-20 beta users onboarded
- [ ] Feedback collected
- [ ] Critical issues fixed

**Week 7: Production Prep**
- [ ] Production API built (FastAPI + Celery)
- [ ] Docker + Kubernetes deployment
- [ ] Load testing: 50-100 concurrent requests (voice feature)
- [ ] Monitoring operational

**Week 8: Launch**
- [ ] Public launch
- [ ] Documentation complete
- [ ] Security audit passed
- [ ] Initial users successfully transcribing

### 8.2 Launch Day Success Criteria (End of Week 8)

**Technical (Must-Have):**
- ✅ **WER < 15%** on Malaysian test set
- ✅ Code-switching F1 > 85%
- ✅ Particle recall > 80%
- ✅ RTF < 0.3 on T4 GPU
- ✅ API uptime > 99.0%
- ✅ Can handle 50-100 concurrent requests (voice is secondary, conservative target)

**Product (Must-Have):**
- ✅ `/v1/transcribe` endpoint (sync)
- ✅ `/v1/transcribe/async` endpoint (async)
- ✅ API key authentication
- ✅ Swagger documentation
- ✅ 20-50 beta users validated system

**Infrastructure (Must-Have):**
- ✅ Kubernetes cluster with auto-scaling
- ✅ Monitoring + alerting (Prometheus/Grafana)
- ✅ SSL + domain configured
- ✅ Database backups automated
- ✅ CI/CD pipeline operational

**Nice-to-Have (Can defer to v1.1):**
- ⏭️ WER < 13% (v1.1 goal)
- ⏭️ Real-time streaming API
- ⏭️ Speaker diarization
- ⏭️ Custom vocabulary support
- ⏭️ Mobile SDK

### 8.3 Post-Launch Success (Months 3-4)

**Stabilization Phase (Month 1 post-launch):**
- Monitor system 24/7 for first 2 weeks
- Fix critical bugs within 24 hours
- Collect production usage data
- Optimize infrastructure costs

**Growth Metrics (Months 2-3):**
- 50-200 active users
- 5,000+ minutes transcribed
- WER improvement to < 13% with production data
- 99.5% uptime
- Infrastructure cost < $800/month

---

## 9. Appendix

### 9.1 Daily Sprint Breakdown (Sample: Week 1)

**Day 1 (Monday): Kickoff**
- 9am: Team kickoff meeting
- 10am-12pm: Set up cloud accounts, provision A100
- 2pm-5pm: Install Unsloth, PyTorch, dependencies
- **Deliverable:** GPU ready, baseline Whisper tested

**Day 2-3 (Tue-Wed): Data Sourcing**
- Download Common Voice Malay dataset (20+ hours)
- Explore additional datasets (Malaya-Speech)
- Data quality check and filtering
- **Deliverable:** 15-20 hours curated data

**Day 4-5 (Thu-Fri): Data Preparation**
- Convert to HuggingFace dataset format
- Create train/val/test splits (80/10/10)
- Run automated QC checks
- **Deliverable:** Dataset ready for training

**Weekend (Optional):**
- Review training scripts
- Pre-read Unsloth documentation
- Prepare for Week 2 training sprint

### 9.2 Communication Plan (Lean Team)

**Daily Standup (10 min, async Slack):**
- What did you complete?
- What are you working on today?
- Any blockers? (escalate immediately)

**End-of-Week Sync (30 min, Friday):**
- Demo this week's progress
- Review against milestone
- Plan next week

**Tools:**
- Slack: All communication
- GitHub: Code + documentation
- Google Docs: Shared notes
- Zoom: Video calls (only when needed)

**Decision-Making:**
- ML Engineer decides: Model architecture, hyperparameters
- Backend/DevOps decides: Infrastructure, deployment
- Collaborative: API design, monitoring strategy
- Fast decisions: <24 hour turnaround on any blocker

### 9.3 Quick Reference

**Key Milestones:**
- Day 7: Data ready
- Day 14: Model trained (WER < 20%)
- Day 21: API ready
- Day 28: Production launch

**Budget Checkpoints:**
- Week 1: $5K
- Week 2: $12K
- Week 3: $18K
- Week 4: $30K (total)

**Emergency Contacts:**
- GPU issues: Cloud provider support
- Training issues: Unsloth GitHub Discussions
- Infrastructure: Kubernetes community Slack

### 9.4 Post-Launch Roadmap (v1.1 - v2.0)

**v1.1 (Weeks 5-8): Optimization**
- Improve WER to <17% with production data
- Add WebSocket streaming API
- Optimize costs (switch to T4 or spot instances)
- Implement rate limiting tiers

**v1.2 (Month 3-4): Features**
- Speaker diarization (who said what)
- Punctuation and capitalization
- Custom vocabulary support
- Mobile SDK (iOS/Android)

**v2.0 (Month 5-6): Enterprise**
- On-premise deployment option
- SSO authentication (SAML, OAuth)
- SLA guarantees (99.9% uptime)
- Advanced analytics dashboard

---

## Summary: 8-Week Production Deployment

**Timeline:** 8 weeks from kickoff to public launch  
**Budget:** $50K-$65K (recommended balanced option)  
**Team:** 3 people (ML Engineer, Backend/DevOps, Full-Stack)  
**Goal:** Production-grade Malaysian ASR with WER < 15%

### 8-Week Timeline Overview:

```
Weeks 1-2: Foundation & Data    →  Environment + 30-40hrs data
Weeks 3-5: Training             →  Model trained, WER < 15%
Week 6:    Beta Testing          →  20-50 users, feedback
Week 7:    Production Prep       →  API + K8s deployment
Week 8:    Public Launch         →  Production Live ✨
```

### Key Success Factors:

1. ⏱️ **Balanced Timeline**: 8 weeks allows proper development & testing
2. 🎯 **Quality First**: WER < 15% achievable with sufficient data & training time
3. 📊 **Data Investment**: 30-40 hours Malaysian speech (50% free + 50% custom)
4. 💪 **Professional Team**: 3 dedicated people for 8 weeks
5. ✅ **Beta Testing**: Week 6 validates quality before launch
6. 🔧 **Production-Ready**: Proper monitoring, scaling, security

### Immediate Actions (Week 1, Day 1):

**Morning:**
- [ ] Team kickoff meeting
- [ ] Provision A100 GPU instance
- [ ] Set up GitHub repo + project management

**Afternoon:**
- [ ] Install Unsloth + PyTorch + dependencies
- [ ] Test baseline Whisper on sample Malaysian audio
- [ ] Start voice actor recruitment

**End of Day 1:**
- [ ] GPU environment operational
- [ ] Baseline WER measured on Malaysian test set
- [ ] Voice actor job postings live

### Budget Allocation:

| Category | Amount | Notes |
|----------|--------|-------|
| Personnel (8 weeks) | $64K | 3 people full-time |
| Infrastructure | $11K | A100 training + T4 inference |
| Data Collection | $3.5K | Voice actors + QA |
| Contingency | $8K | 10% buffer |
| **Total** | **~$87K** | **Recommended: $50-65K** |

### Questions?
- **Technical Issues**: GitHub Issues or Unsloth Discord
- **Project Questions**: Team lead / Slack channel
- **Documentation**: See other 6 docs in `/asr/docs/`

---

**Let's build production-grade Malaysian ASR in 8 weeks! 🚀🇲🇾**

*Timeline: Realistic and achievable with quality focus*  
*Cost-optimized: Record at 48kHz/24-bit to enable TTS reuse!*  
*Last updated: October 12, 2025*

---

**💡 Pro Tip:** By recording at 48kHz/24-bit, your voice data can be reused for TTS training in Months 3-4, saving $6,000-10,000 on the TTS project! See [08_Shared_Data_Strategy.md](08_Shared_Data_Strategy.md) for complete details.

