# Project Execution Plan & Timeline
# Malaysian Multilingual TTS System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** Project Management Team

---

## 1. Executive Summary

### 1.1 Project Overview

**Project Name:** Malaysian Multilingual TTS  
**Duration:** 8 weeks (balanced quality & speed)  
**Budget:** $55K-$75K (balanced investment)  
**Team Size:** 3-4 people (focused team)  
**Success Criteria:** Launch production-ready TTS with MOS > 4.0, scalable architecture

**Key Objectives:**
- ⏱️ **Reasonable timeline**: 8 weeks allows proper fine-tuning & testing
- 🎯 **Fine-tune existing models**: Use XTTS v2 or StyleTTS2 as base
- 💰 **Balanced budget**: Professional voice actors + quality data
- 📊 **Quality target**: MOS > 4.0 (achievable with 8 weeks)
- ✅ **Beta testing**: Time for user validation before launch

### 1.2 Key Milestones (8 Weeks)

```
Weeks 1-2: Foundation & Data    →  Environment + 15-20hrs data
Weeks 3-5: Fine-Tuning          →  Model trained, MOS > 4.0
Week 6:    Beta Testing          →  20-50 users, feedback
Week 7:    Production Prep       →  API + K8s deployment
Week 8:    Public Launch         →  Production Live ✨
```

**Approach:** Fine-tune XTTS v2 (or StyleTTS2 as backup) on 15-25 hours of high-quality Malaysian speech with diverse speakers and scenarios.

---

## 2. Detailed Timeline

### 2.1 Gantt Chart (8 Weeks - Balanced)

```
Weeks 1-2: Foundation & Data Collection
│████████████████████████████████│ Setup, Voice Recording
                                 
Weeks 3-5: Fine-Tuning & Optimization
        │████████████████████████████████████████│ XTTS v2 Fine-tuning
                                                 
Week 6: Beta Testing & Iteration
                        │████████████████│ Beta Users, Feedback
                                         
Week 7: Production Preparation
                                │████████████████│ API, Testing
                                                 
Week 8: Deployment & Launch
                                        │████████████████│ Deploy

Timeline: Week 1────2────3────4────5────6────7────8 (Launch)
```

### 2.2 Week-by-Week Breakdown

**Weeks 1-2: Foundation & Data Collection**
- **Week 1, Days 1-3**: Environment setup (XTTS v2, GPU provisioning)
- **Week 1, Days 4-7**: Recruit 3-5 professional voice actors
- **Week 2, Days 1-5**: Record 15-20 hours of Malaysian speech
- **Week 2, Days 6-7**: Data QA, preprocessing, phoneme alignment
- **Milestone M1**: 15-20 hours high-quality data ready

**Weeks 3-5: Fine-Tuning & Optimization**
- **Week 3**: Start fine-tuning XTTS v2 on Malaysian data
- **Week 4**: Continue training, monitor MOS on validation set
- **Week 5**: Complete training, full evaluation, voice cloning tests
- **Milestone M2**: Model achieves MOS > 4.0, voice cloning working

**Week 6: Beta Testing & Iteration**
- **Days 1-2**: Deploy to staging, create simple demo interface
- **Days 3-5**: Onboard 20-50 beta users, collect MOS feedback
- **Days 6-7**: Iterate on model based on feedback
- **Milestone M3**: Beta testing complete, critical issues fixed

**Week 7: Production Preparation**
- **Days 1-3**: Production API development (FastAPI + async)
- **Days 4-5**: Docker containerization, Kubernetes setup
- **Days 6-7**: Load testing (50-100 concurrent for voice feature), security hardening
- **Milestone M4**: Production infrastructure ready

**Week 8: Deployment & Launch**
- **Days 1-2**: Production deployment, monitoring setup
- **Days 3-4**: Smoke tests, documentation
- **Day 5**: Soft launch to beta users
- **Days 6-7**: Public launch, initial user onboarding
- **Milestone M5**: Production system live

---

## 3. Phase-by-Phase Breakdown (8-Week Timeline)

### 3.1 Weeks 1-2: Data Processing & Environment Setup (Reuses Month 1 Data!)

**Objectives:**
- Set up XTTS v2 fine-tuning environment
- **Process master recordings from ASR project** (no new recording needed!)
- Prepare TTS-specific data format
- Test baseline XTTS v2

**⚡ KEY OPTIMIZATION:** This phase reuses the 30-40 hours of master recordings (48kHz/24-bit) collected during ASR Weeks 1-2 (Month 1). **No new voice recording needed = $6,000-10,000 saved!**

**Critical Tasks:**

| Week | Day | Task | Owner | Output |
|------|-----|------|-------|--------|
| **Week 1** | 1-2 | **GPU Environment Setup** | ML Lead | XTTS environment ready |
| | | • Provision A100 GPU for TTS fine-tuning | ML Lead | GPU operational |
| | | • Install XTTS v2 + dependencies | ML Engineer | XTTS v2 installed |
| | | • Test baseline XTTS v2 on sample text | ML Engineer | Baseline tested |
| **Week 1** | 3-5 | **Process Master Recordings for TTS** | ML Engineer | TTS dataset ready |
| | | • Load master 48kHz recordings (from Month 1) | Data Lead | Master files loaded |
| | | • Downsample to 22.05kHz for TTS | ML Engineer | 22.05kHz audio files |
| | | • Quality check: audio clarity, speaker consistency | Data Lead | QA passed |
| | | • Organize by speaker (3-5 speakers) | ML Engineer | Speaker-separated |
| **Week 1** | 6-7 | **Create TTS Metadata** | ML Engineer | Metadata ready |
| | | • Load transcripts from ASR project (same transcripts!) | Data Lead | Transcripts loaded |
| | | • Create TTS-specific metadata format | ML Engineer | TTS metadata.csv |
| | | • Phoneme alignment (if needed for XTTS v2) | ML Engineer | Alignments ready |
| | | • Validate data: 30-40 hours total @ 22.05kHz | Data Lead | Validation passed |
| **Week 2** | 1-3 | **Prepare Training Splits** | ML Engineer | Splits ready |
| | | • Create per-speaker train/val/test splits | ML Engineer | Speaker splits |
| | | • Ensure code-switching coverage in each split | Data Lead | Diversity checked |
| | | • Create shared multi-speaker test set (200 samples) | ML Engineer | Shared test set |
| **Week 2** | 4-5 | **Baseline Testing** | ML Lead | Baseline MOS |
| | | • Test baseline XTTS v2 on Malaysian text | ML Engineer | Baseline MOS measured |
| | | • Identify quality gaps vs target | ML Lead | Gap analysis |
| | | • Plan fine-tuning strategy | ML Lead | Training plan ready |
| **Week 2** | 6-7 | **Final Preparation** | All | Ready to train |
| | | • Verify all data processed correctly | Data Lead | ✅ Data validated |
| | | • Set up training scripts | ML Engineer | Scripts ready |
| | | • Configure training parameters | ML Lead | Config ready |
| | | • Final checklist for Week 3 training start | All | ✅ Milestone M1 |

**Deliverables:**
- ✅ **GPU environment operational** (A100 + XTTS v2)
- ✅ **Baseline XTTS v2 tested** on Malaysian text samples
- ✅ **30-40 hours TTS-ready data** at 22.05kHz:
  - 3-5 diverse speakers (same speakers as ASR!)
  - Code-switching scenarios (Malay + English)
  - Discourse particles included
  - **Processed from Month 1 master recordings (no new recording!)**
- ✅ **TTS metadata ready:** speaker IDs, transcripts, durations, alignments
- ✅ **Training splits created:** train/val/test per speaker
- ✅ **Baseline MOS measured** (expect ~3.0-3.5 before fine-tuning)

**Budget (Weeks 1-2):** $8,000-$11,000
- ~~Voice actors: $0~~ (reusing Month 1 recordings!) ✅ **$8,000 saved!**
- ~~Studio recording: $0~~ (reusing Month 1 recordings!) ✅ **$2,000 saved!**
- GPU (A100): $2,000 (80 hours × $25/hr for processing & baseline)
- Salaries (3 people × 2 weeks): $9,000
- Tools & licenses: $500

**Key Decisions:**

| Decision | Rationale |
|----------|-----------|
| **Reuse ASR recordings** | Master 48kHz recordings are perfect for TTS (just downsample to 22.05kHz) |
| **Same speakers** | Consistency across ASR and TTS, same voice characteristics |
| **Same transcripts** | Already validated during ASR data prep (99%+ accuracy) |
| **XTTS v2 (confirmed)** | Best for voice cloning, supports multilingual naturally |
| **22.05kHz sampling** | Standard for TTS, good quality vs file size balance |

**⚡ COST SAVINGS:** $6,000-10,000 saved by reusing Month 1 recordings!  
**⏰ TIME SAVINGS:** 2 weeks of voice actor scheduling, recording, and QA eliminated!

---

### 3.2 Weeks 3-5: Fine-Tuning & Optimization

**Objectives:**
- Fine-tune XTTS v2 on Malaysian data
- Achieve MOS > 4.0
- Test voice cloning with 3-second samples

**Critical Tasks:**

| Week | Task | Owner | Output |
|------|------|-------|--------|
| **Week 3** | Start fine-tuning, configure hyperparameters | ML Lead | Training started |
| **Week 4** | Monitor training, adjust learning rate | ML Engineer | Epoch 2-3 complete |
| **Week 5** | Complete training, full evaluation, voice cloning | ML Lead | Model ready, MOS > 4.0 |

**Deliverables:**
- ✅ Fine-tuned XTTS v2 model
- ✅ MOS > 4.0 on Malaysian test set
- ✅ Voice cloning working (3-5 second samples)
- ✅ Code-switching natural and accurate
- ✅ Discourse particles pronounced correctly
- ✅ Model exported and optimized (quantization)

**Budget (Weeks 3-5):** $22,000-$28,000
- GPU (A100): $9,000 (360 hours × $25/hr for fine-tuning)
- Salaries (3 people × 3 weeks): $13,500
- Evaluation (MOS testing): $1,500
- Tools: $500

**Key Metrics:**
- MOS (Mean Opinion Score): > 4.0
- Voice similarity (speaker embedding): > 0.85
- Code-switching naturalness: > 90%
- RTF (Real-Time Factor): < 0.5

---

### 3.3 Week 6: Beta Testing & Iteration

**Objectives:**
- Deploy to staging environment
- Onboard 20-50 beta users
- Collect feedback and iterate

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **Days 1-2** | Deploy to staging, create demo interface | Backend Eng | Staging live |
| **Days 3-5** | Onboard beta users, collect MOS ratings | Product Lead | Feedback data |
| **Days 6-7** | Analyze feedback, iterate on model | ML Lead | Issues fixed |

**Deliverables:**
- ✅ Staging environment deployed
- ✅ 20-50 beta users tested system
- ✅ MOS feedback collected (target: > 4.0)
- ✅ Critical issues identified and fixed
- ✅ User satisfaction > 80%

**Budget (Week 6):** $5,000-$7,000
- GPU (staging): $500
- Salaries (3 people × 1 week): $4,500
- Beta user incentives: $500
- Tools: $200

---

### 3.4 Week 7: Production Preparation

**Objectives:**
- Build production API (FastAPI)
- Docker + Kubernetes deployment
- Load testing and security hardening

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **Days 1-3** | Production API development (FastAPI + async) | Backend Eng | API ready |
| **Days 4-5** | Docker containerization, K8s setup | DevOps | Deployment ready |
| **Days 6-7** | Load testing (50-100 concurrent for voice), security audit | All | Production validated |

**Deliverables:**
- ✅ Production API (FastAPI + async endpoints)
- ✅ Docker images built and tested
- ✅ Kubernetes deployment configured
- ✅ Load tested: 50-100 concurrent requests (voice feature)
- ✅ Security audit passed (basic)
- ✅ Monitoring operational (Prometheus/Grafana)

**Budget (Week 7):** $7,000-$9,000
- GPU (production testing): $1,000
- Salaries (3 people × 1 week): $4,500
- K8s cluster: $800
- Security tools: $500
- Load testing: $200

---

### 3.5 Week 8: Deployment & Launch

**Objectives:**
- Production deployment
- Public launch
- Initial user onboarding

**Critical Tasks:**

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| **Days 1-2** | Production deployment, monitoring setup | DevOps | System live |
| **Days 3-4** | Smoke tests, documentation finalization | All | QA passed |
| **Day 5** | Soft launch to beta users | Product Lead | Validation |
| **Days 6-7** | Public launch, marketing, user support | All | Launch complete |

**Deliverables:**
- ✅ Production system live and monitored
- ✅ Public domain configured (api.tts.example.com)
- ✅ Documentation complete (API docs, user guides)
- ✅ Initial users successfully generating speech
- ✅ Monitoring dashboards operational
- ✅ Support channels established

**Budget (Week 8):** $6,000-$8,000
- GPU (production): $1,500
- Salaries (3 people × 1 week): $4,500
- Marketing: $1,000
- Domain + SSL: $100
- Tools: $200

---

## 4. Budget Estimates (8-Week Timeline)

### 4.1 Budget Breakdown

**Personnel Costs (8 Weeks):**

| Role | Weekly Rate | Weeks | Total |
|------|-------------|-------|-------|
| ML Engineer (Lead) | $3,000 | 8 | $24,000 |
| Backend/DevOps Engineer | $2,800 | 8 | $22,400 |
| Full-Stack Engineer (0.75 FTE) | $2,500 | 8 | $15,000 |
| Audio Engineer (Week 2 only) | $2,000 | 1 | $2,000 |
| **Subtotal** | | | **$63,400** |

**Infrastructure (8 Weeks):**

| Item | Cost | Notes |
|------|------|-------|
| Training GPU (A100, 360 hours) | $9,000 | Weeks 3-5: $25/hr × 360 hrs |
| Production GPU (T4, 4 weeks) | $2,000 | Weeks 5-8: $0.50/hr × 4032 hrs |
| Kubernetes cluster (GKE/EKS) | $800 | Control plane + nodes, 8 weeks |
| Database (PostgreSQL) | $160 | Managed service |
| Storage (S3/GCS) | $100 | 500GB for audio |
| Domain + SSL | $100 | Domain + cert |
| **Subtotal** | | **$12,160** |

**Data Collection:**
| Item | Cost | Notes |
|------|------|-------|
| ~~Voice actors (15-20hrs)~~ | ~~$8,000~~ $0 | ✅ Reusing ASR Month 1 recordings! |
| ~~Studio recording~~ | ~~$2,000~~ $0 | ✅ Reusing ASR Month 1 recordings! |
| Audio processing & TTS formatting | $500 | Downsample 48kHz→22.05kHz, create TTS metadata |
| **Subtotal** | | **$500** (-$10,500 saved!) |

**Evaluation & Testing:**
| Item | Cost | Notes |
|------|------|-------|
| MOS testing (Week 5-6) | $1,500 | 20-50 evaluators |
| Beta user incentives | $500 | Feedback rewards |
| Load testing tools | $200 | Performance testing |
| **Subtotal** | | **$2,200** |

**Tools & Services:**
| Item | Cost | Notes |
|------|------|-------|
| Monitoring (Prometheus/Grafana) | $0 | Self-hosted |
| GitHub Actions | $0 | Free tier |
| Development tools & libraries | $300 | Audio tools, etc. |
| **Subtotal** | | **$300** |

**Contingency (10% for unknowns):** $7,806

**TOTAL 8-WEEK BUDGET:** **$85,866** (was $97,966, saved $12,100 via shared data!)

### 4.2 Budget Options for 8-Week Timeline

**Option 1: Lean MVP ($30K-$40K)** 
- 2-3 people (ML Engineer + Full-Stack, part-time DevOps)
- RTX 4090 or reduced A100 hours
- ✅ Reuses ASR voice data (no new recording!)
- Self-host or use free-tier Kubernetes
- Target: MOS > 3.8, basic voice cloning

**Option 2: Balanced Production ($45K-$60K)** ← **Recommended**
- 3 people full-time
- A100 for fine-tuning, T4 for inference
- ✅ Reuses 30-40 hours from ASR project (no new recording!)
- Managed Kubernetes (GKE/EKS)
- Full monitoring, beta testing phase
- Target: MOS > 4.0, production-grade voice cloning

**Option 3: Premium Quality ($75K-$100K)**
- 4 people including project lead
- Best hardware (A100 + A10 inference)
- ✅ Reuses ASR data + optional 10-20 more hours
- Enterprise-grade infrastructure
- Extensive testing and documentation
- Target: MOS > 4.2, enterprise-ready

**Comparison:**

| Item | Lean | Balanced | Premium |
|------|------|----------|---------|
| **Budget** | $30-40K | $45-60K | $75-100K |
| **Team** | 2-3 people | 3 people | 4 people |
| **Timeline** | 10 weeks | 8 weeks | 8 weeks |
| **MOS Target** | > 3.8 | > 4.0 | > 4.2 |
| **Voice Data** | Reuse ASR (30hrs) | Reuse ASR (30-40 hrs) | Reuse ASR + 10-20 more |
| **Speakers** | Same as ASR (3-5) | Same as ASR (3-5) | ASR + new (5-7) |
| **Infrastructure** | Basic | Production | Enterprise |
| **Data Collection Cost** | **$0 (shared!)** ✅ | **$0 (shared!)** ✅ | $0-3K (optional more) |
| **Risk** | Medium | Low | Very Low |

---

## 5. Risk Management (8-Week Timeline)

### 5.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **MOS doesn't reach > 4.0** | Low-Medium | High | More data (25+ hours), longer fine-tuning, try StyleTTS2 |
| **Voice actor delays** | Medium | Medium | Recruit Week 1, have backup actors, reduce data to 12hrs |
| **XTTS v2 fine-tuning issues** | Low | High | StyleTTS2 as backup, community support, pre-test Week 1 |
| **Voice cloning quality poor** | Low-Medium | Medium | More speaker diversity, longer voice samples |
| **Team member unavailable** | Low | Medium | 8-week buffer, cross-training, documentation |
| **Infrastructure issues** | Low | Medium | Test early, multi-cloud setup |
| **Scope creep** | Low | Low | 8 weeks allows flexibility, strict MVP focus |
| **Security vulnerabilities** | Low | High | Week 7 security audit, penetration testing |

### 5.2 Mitigation Strategies (Balanced Approach)

**For MOS Target:**
- **Primary**: XTTS v2 with 15-25 hours quality data
- **Backup Plan**: StyleTTS2 if XTTS v2 issues arise
- **Target**: MOS > 4.0 achievable with proper data & training

**For Data Collection:**
- **Week 1**: Start recruitment immediately
- **Week 2**: Record with professional setup
- **Quality**: 95%+ audio clarity, native speakers only

**For Timeline:**
- **Weekly milestones**: Clear checkpoints
- **Parallel work**: Data recording || Environment setup
- **Buffer time**: 8 weeks allows 1-2 week contingency

**For Team Capacity:**
- **Cross-training**: Everyone understands components
- **Documentation**: Comprehensive from day 1
- **Knowledge sharing**: Weekly demos

---

## 6. Decision Gates (8-Week Timeline)

### 6.1 Go/No-Go Criteria

**Gate 1: End of Week 2**
- ✅ GPU environment operational (A100 + XTTS v2)
- ✅ Baseline XTTS v2 tested on Malaysian samples
- ✅ 15-25 hours voice data collected and preprocessed
- ✅ Data quality validated (95%+ clarity)
- **Decision:** Proceed to fine-tuning OR collect more data

**Gate 2: End of Week 4 (Mid-Training)**
- ✅ Fine-tuning progressing (epoch 2-3)
- ✅ Loss decreasing steadily
- ✅ Preliminary MOS > 3.5 on validation set
- ✅ Voice cloning working with test samples
- **Decision:** Continue training OR adjust hyperparameters

**Gate 3: End of Week 5 (Model Complete)**
- ✅ MOS > 4.0 on test set
- ✅ Voice cloning quality high (similarity > 0.85)
- ✅ Code-switching natural (> 90%)
- ✅ Discourse particles correct
- **Decision:** Proceed to beta OR extend training 1 week

**Gate 4: End of Week 6 (Beta Testing)**
- ✅ 20-50 beta users onboarded
- ✅ Feedback collected (MOS, satisfaction)
- ✅ Critical issues identified
- ✅ User satisfaction > 80%
- **Decision:** Proceed to production OR iterate

**Gate 5: End of Week 7 (Production Ready)**
- ✅ Production API functional
- ✅ Load tested (50-100 concurrent for voice feature)
- ✅ Docker + K8s deployment working
- ✅ Monitoring operational
- **Decision:** Launch Week 8 OR delay for critical fixes

**Gate 6: Week 8 Launch Decision**
- ✅ All systems tested
- ✅ Documentation complete
- ✅ Beta users satisfied
- ✅ Security audit passed
- **Decision:** Public launch OR soft launch

**Contingency Plans:**
- If MOS < 3.8 at Week 5 → Collect 5-10 more hours, extend 1 week
- If voice cloning poor → More speaker samples, adjust embeddings
- If beta feedback critical → Extend beta 1 week

---

## 7. Success Criteria (8-Week Launch)

### 7.1 Week-by-Week Success Metrics

**Week 1: Environment Setup**
- [ ] GPU environment ready
- [ ] XTTS v2 installed and tested
- [ ] Voice actors recruited (3-5 people)

**Week 2: Data Collection**
- [ ] 15-25 hours voice data recorded
- [ ] Audio quality: 95%+ clarity
- [ ] Data preprocessed and aligned

**Weeks 3-4: Fine-Tuning (Part 1)**
- [ ] Fine-tuning started
- [ ] Epoch 1-2 complete
- [ ] Validation MOS > 3.5

**Week 5: Training Complete**
- [ ] **MOS > 4.0** on test set
- [ ] Voice cloning working (3-sec samples)
- [ ] Code-switching natural
- [ ] Model exported and optimized

**Week 6: Beta Testing**
- [ ] Staging deployed
- [ ] 20-50 beta users onboarded
- [ ] Feedback collected
- [ ] User satisfaction > 80%

**Week 7: Production Prep**
- [ ] Production API built
- [ ] Docker + K8s deployment
- [ ] Load testing: 50-100 concurrent (voice feature)
- [ ] Security audit passed

**Week 8: Launch**
- [ ] Public launch
- [ ] Documentation complete
- [ ] Initial users generating speech
- [ ] Monitoring operational

### 7.2 Launch Day Success Criteria (End of Week 8)

**Technical (Must-Have):**
- ✅ **MOS > 4.0** on Malaysian test set
- ✅ Voice cloning working (3-5 second samples)
- ✅ Voice similarity > 0.85
- ✅ Code-switching natural (> 90%)
- ✅ RTF < 0.5 on T4 GPU
- ✅ API uptime > 99.0%
- ✅ Can handle 50-100 concurrent requests (voice is secondary feature)

**Product (Must-Have):**
- ✅ `/v1/synthesize` endpoint (text-to-speech)
- ✅ `/v1/clone` endpoint (voice cloning)
- ✅ API key authentication
- ✅ Swagger documentation
- ✅ 20-50 beta users validated

**Infrastructure (Must-Have):**
- ✅ Kubernetes cluster with auto-scaling
- ✅ Monitoring + alerting
- ✅ SSL + domain configured
- ✅ Database backups automated
- ✅ CI/CD pipeline operational

**Nice-to-Have (Can defer to v1.1):**
- ⏭️ MOS > 4.2 (v1.1 goal)
- ⏭️ Real-time streaming TTS
- ⏭️ Custom voice training (user uploads)
- ⏭️ Emotion control
- ⏭️ Mobile SDK

### 7.3 Post-Launch Success (Months 3-4)

**Stabilization Phase (Month 1 post-launch):**
- Monitor 24/7 for first 2 weeks
- Fix critical bugs within 24 hours
- Collect production usage data
- Optimize infrastructure costs

**Growth Metrics (Months 2-3):**
- 50-200 active users
- 10,000+ audio generations
- MOS improvement to > 4.2 with production data
- 99.5% uptime
- Infrastructure cost < $1,000/month

---

## 8. Summary: 8-Week TTS Production Deployment

**Timeline:** 8 weeks from kickoff to public launch  
**Budget:** $45K-$60K (recommended balanced option - $10-15K less via shared data!)  
**Team:** 3 people (ML Engineer, Backend/DevOps, Full-Stack - same team from ASR!)  
**Goal:** Production-grade Malaysian TTS with MOS > 4.0

### 8-Week Timeline Overview:

```
Weeks 1-2: Data Processing Only →  Reuse ASR Month 1 recordings (no new recording!)
Weeks 3-5: Fine-Tuning          →  XTTS v2 trained, MOS > 4.0
Week 6:    Beta Testing          →  20-50 users, feedback
Week 7:    Production Prep       →  API + K8s deployment
Week 8:    Public Launch         →  Production Live ✨
```

### Key Success Factors:

1. ⚡ **Shared Data Strategy**: Reuses ASR recordings = $10K saved, no new recording needed!
2. ⏱️ **Balanced Timeline**: 8 weeks allows proper fine-tuning & testing
3. 🎯 **Quality First**: MOS > 4.0 achievable with 30-40hrs high-quality data
4. 🎤 **Professional Audio**: Same 48kHz master recordings from ASR (downsampled to 22.05kHz)
5. 💪 **Same Team**: Smooth transition from ASR to TTS (no new hiring)
6. ✅ **Beta Validation**: Week 6 ensures quality before launch
7. 🔧 **Production-Ready**: Proper monitoring, scaling, security

### Immediate Actions (Week 1, Day 1):

**Morning:**
- [ ] Team transition meeting (from ASR to TTS)
- [ ] Provision A100 GPU instance for TTS
- [ ] Set up TTS training environment

**Afternoon:**
- [ ] Install XTTS v2 + dependencies
- [ ] Test baseline XTTS v2 on sample Malaysian text
- [ ] Load master recordings from ASR project (Month 1)

**End of Day 1:**
- [ ] GPU environment operational
- [ ] Baseline XTTS v2 tested
- [ ] Master recordings (48kHz) accessible and validated

### Budget Allocation:

| Category | Amount | Notes |
|----------|--------|-------|
| Personnel (8 weeks) | $63K | 3 people (same team from ASR) |
| Infrastructure | $12K | A100 training + T4 inference |
| ~~Voice Data Collection~~ | ~~$11K~~ **$0.5K** | ✅ **Reuses ASR data! $10.5K saved!** |
| Evaluation & Testing | $2K | MOS testing, beta |
| Contingency | $8K | 10% buffer |
| **Total** | **~$86K** | **Recommended: $45-60K** |

**💰 COST SAVINGS vs Original Plan:**
- Original TTS budget: $97K
- Optimized with shared data: $86K
- **Savings: $11K (11% reduction!)**

### Questions?
- **Technical Issues**: XTTS v2 GitHub or Discord
- **Project Questions**: Team lead / Slack channel
- **Documentation**: See other 6 docs in `/tts/docs/`

---

**Let's build production-grade Malaysian TTS in 8 weeks! 🚀🇲🇾**

*Timeline: Realistic and achievable with quality focus*  
*Cost-optimized: Reuses ASR data, saves $10K+ on recording!*  
*Last updated: October 12, 2025*

---

**💡 Pro Tip:** This TTS project benefits from the shared data strategy! See [ASR 08_Shared_Data_Strategy.md](../../asr/docs/08_Shared_Data_Strategy.md) for complete details on how one recording session serves both ASR and TTS.

---

## OLD SECTIONS BELOW (To be removed/archived)

### Phase 2: Data Collection (Weeks 3-14)

#### Weeks 3-4: Voice Actor Recruitment

**Objectives:**
- Recruit 5-10 voice actors
- Finalize recording scripts
- Set up recording infrastructure

**Tasks:**
```
□ Post recruitment ads
□ Screen applicants
□ Conduct auditions
□ Sign contracts
□ Prepare recording scripts (10,000 sentences)
□ Set up recording equipment
```

**Team:**
- Project Manager (1)
- Audio Engineer (1)
- Linguist (1)

**Budget:** $5,000 (recruitment + setup)

#### Weeks 5-12: Recording Sessions

**Objectives:**
- Record 50-75 hours of speech
- 10-15 hours per speaker

**Schedule:**
| Week | Speaker | Hours | Status |
|------|---------|-------|--------|
| 5-6  | SP001, SP002 | 20 | Scheduled |
| 7-8  | SP003, SP004 | 20 | Scheduled |
| 9-10 | SP005, SP006 | 20 | Scheduled |
| 11-12| SP007-SP010 | 15 | Scheduled |

**Team:**
- Audio Engineer (1)
- Studio Assistant (1)
- Project Manager (0.5)

**Budget:** $30,000
- Voice actors: $25,000
- Studio rental: $3,000
- Equipment: $2,000

#### Weeks 8-14: Annotation & QC

**Objectives:**
- Transcribe and annotate all recordings
- Language tagging
- Forced alignment
- Quality control

**Tasks:**
```
Week 8-9:  Transcription
Week 10-11: Language tagging
Week 12-13: Forced alignment (MFA)
Week 14:    Quality control and validation
```

**Team:**
- Annotators (3)
- Data Engineer (1)
- QA Specialist (1)

**Budget:** $15,000

**Deliverable:** Annotated dataset ready for training

---

### Phase 3: Infrastructure Setup (Weeks 6-10)

#### Weeks 6-7: Cloud Infrastructure

**Objectives:**
- Set up AWS/GCP accounts
- Deploy training infrastructure
- Configure storage and databases

**Tasks:**
```
□ Cloud account setup
□ VPC and networking configuration
□ GPU instances (4× p3.2xlarge)
□ S3/storage buckets
□ RDS PostgreSQL
□ ElastiCache Redis
□ IAM roles and permissions
```

**Team:**
- DevOps Engineer (1)
- ML Engineer (1)

**Budget:** $2,000 (setup) + $2,500/month (ongoing)

#### Weeks 8-9: Training Pipeline

**Objectives:**
- Build data preprocessing pipeline
- Set up training scripts
- Implement logging and monitoring

**Deliverables:**
- Data preprocessing code
- Training scripts
- Experiment tracking (W&B/MLflow)

**Team:**
- ML Engineers (2)
- Data Engineer (1)

#### Week 10: Monitoring Setup

**Objectives:**
- Set up Prometheus + Grafana
- Configure alerting
- Create dashboards

**Team:**
- DevOps Engineer (1)
- ML Engineer (1)

---

### Phase 4: Model Development (Weeks 9-28)

#### Weeks 9-12: Baseline Model

**Objectives:**
- Implement FastSpeech 2 baseline
- Train on single language
- Validate pipeline

**Milestones:**
- Week 9: Model implementation complete
- Week 10: First training run
- Week 11: Baseline results
- Week 12: Evaluation and iteration

**Team:**
- ML Engineers (2)
- Research Scientist (1)

**Budget:** Cloud compute: $10,000

**Success Criteria:**
- Model trains without errors
- MOS > 3.5 on single language
- Pipeline validated

#### Weeks 13-18: Pre-training (Optional)

**Objectives:**
- Pre-train on high-resource languages
- Transfer learning setup

**Data:**
- LJSpeech (English)
- AISHELL-3 (Mandarin)
- VCTK (multi-speaker English)

**Target Metrics:**
- MCD < 7.0 dB
- Loss convergence

**Team:**
- ML Engineers (2)

**Budget:** Cloud compute: $15,000

#### Weeks 17-26: Main Training

**Objectives:**
- Train on full Malaysian dataset
- Implement code-switching
- Curriculum learning

**Curriculum:**
```
Week 17-19: Single-language utterances
Week 20-23: Two-language code-switching
Week 24-26: Full code-switching
```

**Target Metrics:**
- MOS > 3.8
- Code-switching accuracy > 90%
- MCD < 6.5 dB

**Team:**
- ML Engineers (2)
- Research Scientist (1)

**Budget:** Cloud compute: $25,000

**Weekly Check-ins:**
- Monitor training loss
- Generate sample audio
- Evaluate on validation set

#### Weeks 25-28: Fine-tuning

**Objectives:**
- Fine-tune on high-quality subset
- Optimize particle pronunciation
- Final model optimization

**Target Metrics:**
- MOS > 4.0
- Particle quality > 4.2

**Team:**
- ML Engineers (2)

**Budget:** Cloud compute: $10,000

---

### Phase 5: Vocoder Training (Weeks 20-28)

**Parallel to main training**

**Objectives:**
- Train HiFi-GAN vocoder
- Optimize for quality and speed

**Timeline:**
```
Week 20-23: Initial training (ground-truth mels)
Week 24-26: Fine-tuning (predicted mels)
Week 27-28: Optimization and quantization
```

**Target Metrics:**
- MOS > 4.0
- RTF < 0.1 (vocoder only)
- No artifacts

**Team:**
- ML Engineer (1)
- Audio Engineer (0.5)

**Budget:** Cloud compute: $12,000

---

### Phase 6: Evaluation (Weeks 29-36)

#### Weeks 29-32: Objective Evaluation

**Objectives:**
- Comprehensive objective metrics
- Automated evaluation pipeline

**Metrics:**
- MCD
- F0 RMSE
- Duration MAE
- WER (ASR round-trip)
- PESQ, STOI

**Team:**
- ML Engineer (1)
- QA Engineer (1)

**Budget:** $2,000

#### Weeks 31-36: Subjective Evaluation

**Objectives:**
- MOS testing (30 raters)
- Code-switching quality evaluation
- Particle quality assessment

**Timeline:**
```
Week 31-32: Rater recruitment and training
Week 33-35: MOS testing sessions
Week 36:    Analysis and reporting
```

**Team:**
- Research Scientist (1)
- Project Manager (0.5)
- Raters (30)

**Budget:** $8,000
- Rater compensation: $6,000
- Platform/tools: $2,000

**Deliverable:** Evaluation report with decision to proceed

---

### Phase 7: Optimization (Weeks 33-38)

**Parallel to evaluation**

#### Weeks 33-38: Model Optimization

**Objectives:**
- Quantization (INT8)
- ONNX export
- TensorRT optimization (if using NVIDIA)

**Target:**
- 2-3x speedup
- Minimal quality loss (MOS drop < 0.1)

**Team:**
- ML Engineer (1)
- DevOps Engineer (1)

#### Weeks 35-38: API Development

**Objectives:**
- Build production API
- Implement caching
- Rate limiting
- Authentication

**Deliverables:**
- FastAPI service
- API documentation
- Client SDKs (Python, JavaScript)

**Team:**
- Backend Engineer (1)
- ML Engineer (1)

**Budget:** $10,000

---

### Phase 8: Beta Testing (Weeks 39-44)

#### Weeks 39-40: Beta Launch

**Objectives:**
- Deploy to staging environment
- Invite 50-100 beta users
- Collect initial feedback

**Tasks:**
```
□ Deploy to staging
□ Invite beta users
□ Provide documentation
□ Set up feedback channels
□ Monitor usage
```

**Team:**
- Full team on standby

**Budget:** $5,000
- Staging infrastructure: $3,000
- Beta user incentives: $2,000

#### Weeks 40-44: Feedback & Iteration

**Objectives:**
- Collect detailed feedback
- Fix critical bugs
- Iterate on features

**Weekly Goals:**
- 70% of beta users active
- < 5% critical bugs
- NPS > 40

**Team:**
- Backend Engineer (1)
- ML Engineer (1)
- Product Manager (1)

**Budget:** $8,000

---

### Phase 9: Production Preparation (Weeks 45-48)

#### Week 45-46: Infrastructure Scaling

**Objectives:**
- Set up production environment
- Configure auto-scaling
- Load balancer setup
- CDN configuration

**Tasks:**
```
□ Production Kubernetes cluster
□ Auto-scaling policies
□ Load balancer (ALB)
□ CloudFlare/CDN setup
□ Database replication
□ Backup systems
□ Monitoring and alerting
```

**Team:**
- DevOps Engineer (1)
- Backend Engineer (1)

**Budget:** $10,000

#### Week 45-46: Security Audit

**Objectives:**
- Security penetration testing
- Compliance review
- Privacy audit

**External Service:** Hire security firm

**Budget:** $8,000

#### Week 47: Load Testing

**Objectives:**
- Stress test system
- Identify bottlenecks
- Validate auto-scaling

**Scenarios:**
- Normal load: 100 req/s
- Peak load: 500 req/s
- Spike test: 1000 req/s for 5 minutes

**Team:**
- DevOps Engineer (1)
- Backend Engineer (1)

#### Week 48: Final QA

**Objectives:**
- End-to-end testing
- Disaster recovery drill
- Documentation review
- Launch checklist

**Team:**
- Full team

---

### Phase 10: Launch (Weeks 49-52)

#### Week 49: Canary Deployment

**Objectives:**
- Deploy to 5% of traffic
- Monitor metrics closely
- Validate performance

**Monitoring:**
- Latency (p50, p95, p99)
- Error rate
- User feedback
- System health

**Go/No-Go Criteria:**
- Error rate < 1%
- Latency p95 < 500ms
- No critical bugs

**Team:**
- Full team on standby

#### Week 50: Full Launch

**Objectives:**
- Increase to 100% traffic
- Marketing announcement
- Press release
- Product Hunt launch

**Activities:**
```
□ DNS cutover to production
□ Marketing email blast
□ Social media campaign
□ Product Hunt submission
□ Press release distribution
□ Blog post publication
□ Partnership announcements
```

**Team:**
- Full team
- Marketing team

**Budget:** $10,000 (marketing)

#### Weeks 50-52: Post-Launch Monitoring

**Objectives:**
- Intensive monitoring
- Rapid bug fixes
- User support
- Collect feedback

**Daily Activities:**
- Review metrics dashboard
- Check error logs
- Respond to user feedback
- Fix critical bugs

**Weekly Activities:**
- Team retrospective
- Metrics review
- Feature prioritization
- Documentation updates

**Team:**
- Full team

---

## 4. Resource Allocation

### 4.1 Team Structure

#### Core Team (Full-Time)

| Role | Count | Months | Monthly Rate | Total Cost |
|------|-------|--------|--------------|------------|
| Technical Lead / Senior ML Engineer | 1 | 12 | $12,000 | $144,000 |
| ML Engineers | 2 | 12 | $10,000 | $240,000 |
| Backend Engineer | 1 | 8 | $9,000 | $72,000 |
| Data Engineer | 1 | 6 | $8,500 | $51,000 |
| DevOps Engineer | 1 | 8 | $9,000 | $72,000 |
| Project Manager | 1 | 12 | $8,000 | $96,000 |
| **Subtotal** | **8** | | | **$675,000** |

#### Extended Team (Part-Time/Contract)

| Role | Cost |
|------|------|
| Linguist / Language Expert | $15,000 |
| Audio Engineer | $8,000 |
| Voice Actors (5-10) | $25,000 |
| Annotators (3) | $15,000 |
| QA/Testers | $10,000 |
| MOS Raters (30) | $6,000 |
| Security Auditor | $8,000 |
| **Subtotal** | **$87,000** |

#### Total Personnel: $762,000

### 4.2 Infrastructure Costs

| Category | Monthly | Months | Total |
|----------|---------|--------|-------|
| GPU Instances (4× p3.2xlarge) | $10,000 | 10 | $100,000 |
| Storage (S3, EBS) | $500 | 12 | $6,000 |
| Database (RDS) | $200 | 12 | $2,400 |
| Cache (Redis) | $150 | 12 | $1,800 |
| Networking & CDN | $300 | 12 | $3,600 |
| Monitoring Tools | $200 | 12 | $2,400 |
| **Total Infrastructure** | | | **$116,200** |

### 4.3 Other Costs

| Category | Cost |
|----------|------|
| Software Licenses | $5,000 |
| Office & Equipment | $8,000 |
| Marketing & Launch | $15,000 |
| Legal & Compliance | $5,000 |
| Contingency (15%) | $135,000 |
| **Total Other** | **$168,000** |

### 4.4 Total Budget Summary

| Category | Cost | Percentage |
|----------|------|------------|
| Personnel | $762,000 | 72% |
| Infrastructure | $116,200 | 11% |
| Other | $168,000 | 16% |
| **TOTAL** | **$1,046,200** | **100%** |

**Budget Range:**
- **Minimum (Lean)**: $250,000 (smaller team, longer timeline)
- **Recommended**: $800,000-1,000,000
- **Comfortable**: $1,000,000-1,200,000

---

## 5. Risk Management

### 5.1 Risk Matrix

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| **Insufficient training data** | High | High | Multiple collection strategies; synthetic data | Data Team |
| **Model quality below target** | Medium | High | Pre-training; expert consultation; iterative improvement | ML Team |
| **Budget overrun** | Medium | Medium | Strict tracking; contingency buffer; scope adjustment | PM |
| **Timeline delays** | Medium | Medium | Agile methodology; buffer time; parallel workstreams | PM |
| **Key personnel departure** | Low | High | Documentation; knowledge sharing; backup resources | Tech Lead |
| **Cloud costs exceed estimates** | Medium | Medium | Cost monitoring; auto-shutdown; optimization | DevOps |
| **Security breach** | Low | High | Security audit; penetration testing; monitoring | DevOps |
| **Poor user adoption** | Medium | High | Beta testing; user research; marketing | PM/Marketing |
| **Competitor launches first** | Low | Medium | Differentiation; speed; quality focus | Leadership |
| **Technical infrastructure issues** | Low | Medium | Redundancy; backups; disaster recovery | DevOps |

### 5.2 Mitigation Strategies

#### For High-Priority Risks:

1. **Insufficient Training Data**
   - **Preventive**: Start data collection early (Week 3)
   - **Contingency**: Crowdsourcing + synthetic data generation
   - **Monitoring**: Weekly data collection reviews

2. **Model Quality Below Target**
   - **Preventive**: Weekly model evaluations; expert reviews
   - **Contingency**: Extended fine-tuning; architecture adjustments
   - **Escalation**: Engage external consultants if needed

3. **Budget Overrun**
   - **Preventive**: Monthly budget reviews; variance analysis
   - **Contingency**: 15% contingency buffer; scope reduction options
   - **Decision Points**: Month 3, 6, 9 reviews

---

## 6. Decision Gates

### 6.1 Gate 1: Data Collection Complete (Week 14)

**Criteria:**
- [ ] 50+ hours of audio recorded
- [ ] 90%+ of data annotated
- [ ] Quality validation passed
- [ ] Dataset statistics meet targets

**Decision:**
- **GO**: Proceed to main training
- **REVIEW**: Additional data collection (2-4 weeks)
- **NO-GO**: Reassess project feasibility

### 6.2 Gate 2: Baseline Model (Week 12)

**Criteria:**
- [ ] Model trains successfully
- [ ] MOS > 3.5 on single language
- [ ] Pipeline validated
- [ ] Team confident in approach

**Decision:**
- **GO**: Proceed to pre-training/main training
- **REVIEW**: Architecture adjustments needed
- **NO-GO**: Major pivot required

### 6.3 Gate 3: Main Training Complete (Week 28)

**Criteria:**
- [ ] MOS > 3.8
- [ ] Code-switching works
- [ ] Particle quality > 4.0
- [ ] Technical metrics met

**Decision:**
- **GO**: Proceed to evaluation
- **REVIEW**: Additional fine-tuning (2-4 weeks)
- **NO-GO**: Major retraining required

### 6.4 Gate 4: Evaluation Results (Week 36)

**Criteria:**
- [ ] MOS > 4.0
- [ ] All objective metrics met
- [ ] Code-switching quality > 4.0
- [ ] Particle quality > 4.2
- [ ] No critical issues

**Decision:**
- **GO**: Proceed to beta launch
- **REVIEW**: Targeted improvements (2-4 weeks)
- **NO-GO**: Major iteration required

### 6.5 Gate 5: Beta Testing (Week 44)

**Criteria:**
- [ ] 70%+ user retention
- [ ] NPS > 40
- [ ] < 5% critical bugs
- [ ] Performance targets met
- [ ] User feedback positive

**Decision:**
- **GO**: Proceed to production launch
- **REVIEW**: Address feedback (1-2 weeks)
- **NO-GO**: Extended beta period

---

## 7. Communication Plan

### 7.1 Internal Communication

#### Daily
- Stand-up meetings (15 min)
- Slack updates on progress/blockers

#### Weekly
- Team sync (1 hour)
- Metrics review
- Risk assessment

#### Monthly
- All-hands meeting
- Stakeholder update
- Budget review
- Timeline assessment

### 7.2 Stakeholder Communication

#### Bi-Weekly
- Status report (email)
- Key metrics dashboard
- Risk updates

#### Monthly
- Executive presentation
- Demo of progress
- Budget vs. actual
- Decision gate reviews

#### Quarterly
- Comprehensive review
- Strategic alignment
- Budget reforecasting

---

## 8. Success Metrics

### 8.1 Project Success Criteria

**Must-Have (Launch Blockers):**
- [ ] MOS Naturalness > 4.0
- [ ] Code-switching functionality working
- [ ] WER < 5%
- [ ] API functional with documentation
- [ ] Security audit passed

**Should-Have:**
- [ ] MOS > 4.2
- [ ] Particle quality > 4.2
- [ ] RTF < 0.3
- [ ] Multi-speaker support
- [ ] 500+ registered users (Month 1)

**Nice-to-Have:**
- [ ] MOS > 4.5
- [ ] Streaming capability
- [ ] Mobile SDK
- [ ] 1000+ registered users (Month 1)

### 8.2 KPIs by Phase

| Phase | KPI | Target |
|-------|-----|--------|
| Data Collection | Hours recorded | 50-75 |
| | Annotation accuracy | > 95% |
| Model Training | Training loss convergence | Stable for 50k steps |
| | Validation MCD | < 6.5 dB |
| Evaluation | MOS Naturalness | > 4.0 |
| | Code-switch Quality | > 4.0 |
| Beta | User retention (Week 1) | > 70% |
| | NPS | > 40 |
| Launch | Uptime | > 99.5% |
| | p95 Latency | < 500ms |
| Post-Launch | MAU growth | 30% MoM |
| | Customer satisfaction | > 4.0/5.0 |

---

## 9. Lessons Learned & Retrospectives

### 9.1 Retrospective Schedule

- After each major phase
- After launch
- Monthly for first 3 months post-launch

### 9.2 Retrospective Format

**What went well?**
**What didn't go well?**
**What should we do differently?**
**Action items**

---

## 10. Conclusion

This execution plan provides a comprehensive roadmap for developing and launching the Malaysian Multilingual TTS system. Success depends on:

1. **Strong Team**: Skilled ML engineers, dedicated data team, solid infrastructure
2. **Quality Data**: 50-75 hours of high-quality, annotated speech
3. **Iterative Approach**: Regular evaluation, feedback, and improvement
4. **Risk Management**: Proactive identification and mitigation
5. **Clear Communication**: Regular updates to all stakeholders

**Key Success Factors:**
- Start data collection early
- Weekly model evaluations
- Don't skip quality gates
- Maintain contingency buffer
- Stay flexible and adapt

**Next Steps:**
1. Secure budget approval
2. Assemble team
3. Kick off data collection
4. Begin infrastructure setup
5. Start weekly progress reviews

---

**Document Version:** 1.0  
**Last Updated:** October 12, 2025  
**Next Review:** Monthly throughout project

