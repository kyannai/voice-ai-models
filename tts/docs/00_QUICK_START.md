# Quick Start Guide
# Malaysian TTS Documentation

**Welcome!** This guide will help you navigate the comprehensive documentation suite for building a Malaysian Text-to-Speech system that synthesizes Malay with English code-switching, Pinyin, slang, and particles.

---

## 📚 Document Overview

I've split the original comprehensive plan into **7 detailed documents**, each focusing on a specific aspect of the project:

### 1. [Product Requirements Document (PRD)](01_PRD_Product_Requirements.md)
- Product vision and market analysis
- User personas and use cases
- Functional and non-functional requirements
- Success metrics and KPIs
- Pricing strategy and competitive analysis

### 2. [Data Preparation Guide](03_Data_Preparation_Guide.md)
- Data collection strategies (voice actors, crowdsourcing, found data)
- Recording specifications and guidelines
- Annotation schema and tools
- Quality control procedures
- Montreal Forced Aligner setup

### 3. [Training Strategy & Guide](04_Training_Strategy_Guide.md)
- XTTS v2 / StyleTTS2 architecture and model selection
- Training pipeline (pre-training → main training → fine-tuning)
- Text processing and phoneme alignment
- Environment setup and requirements
- Curriculum learning strategies
- Vocoder training
- Debugging and optimization tips

### 4. [Evaluation Methodology](05_Evaluation_Methodology.md)
- Objective metrics (MCD, F0, WER, PESQ, STOI)
- Subjective metrics (MOS, preference tests)
- Malaysian-specific evaluation (code-switching, particles)
- Evaluation workflows and protocols
- Production monitoring strategies

### 5. [Deployment Guide](06_Deployment_Guide.md)
- Model inference architecture and optimization
- API design and implementation
- Infrastructure setup (AWS/GCP/On-premise)
- Docker containerization
- Kubernetes deployment
- CI/CD pipeline (GitHub Actions)
- Monitoring (Prometheus, Grafana)
- Security (SSL, rate limiting, authentication)

### 6. [Project Execution Plan & Timeline](07_Project_Execution_Plan.md)
- Detailed 8-week timeline for production deployment
- Phase-by-phase breakdown (data processing → training → deployment)
- Resource allocation and budget ($50K-$65K balanced option)
- Reuses ASR master recordings (saves $10K+!)
- Decision gates and success criteria
- Team structure and roles

### 7. [Scalability Notes](08_Scalability_Notes.md)
- Production-ready architecture for scalable deployment
- Auto-scaling strategies and load balancing
- Performance optimization techniques
- Cost management for cloud infrastructure
- Monitoring and alerting setup

---

## 🔗 Quick Links

- [Main README](../../README.md)
- [01 - Product Requirements](01_PRD_Product_Requirements.md)
- [02 - Technical Architecture](02_Technical_Architecture.md)
- [03 - Data Preparation](03_Data_Preparation_Guide.md)
- [04 - Training Strategy](04_Training_Strategy_Guide.md)
- [05 - Evaluation Methodology](05_Evaluation_Methodology.md)
- [06 - Deployment Guide](06_Deployment_Guide.md)
- [07 - Project Execution Plan](07_Project_Execution_Plan.md)

---

## ✅ Next Steps

1. **Review the document list** above and identify what's relevant for your current task
2. **Start with the PRD** to understand the product vision and requirements
3. **Follow with Technical Architecture** for system overview
4. **Dive into specific guides** (Data Preparation, Training, Deployment) as needed
5. **Bookmark key sections** for quick reference during implementation

---

**Last Updated:** October 12, 2025  
**Document Version:** 1.0

