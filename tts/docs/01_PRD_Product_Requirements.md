# Product Requirements Document (PRD)
# Malaysian TTS System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** TTS Product Team

---

## 1. Executive Summary

### 1.1 Product Vision

Create a natural-sounding Text-to-Speech (TTS) system that authentically reproduces Malaysian speech patterns. The system is optimized for **Malay as the primary language** with natural English code-switching, Pinyin/Mandarin terms, local slang, and proper pronunciation of Malaysian discourse particles.

### 1.2 Problem Statement

Current TTS solutions fail to capture the unique linguistic characteristics of Malaysian speech:
- **No Malay-primary code-switching**: Existing systems treat Malay and English as separate, cannot handle natural Malaysian speech where Malay is primary with English mixed in
- **Lack of particle support**: Critical discourse markers (lah, leh, loh, meh, lor) are mispronounced or ignored
- **Wrong accent**: Standard Malay or English TTS sounds foreign to Malaysian ears
- **Missing slang & special vocab**: Cannot pronounce local terms, Pinyin words, and Malaysian-specific vocabulary
- **Poor prosody**: Rhythm and intonation patterns don't match natural Malaysian speech

### 1.3 Competitive Landscape (to be validated)

| Provider | Strengths | Weaknesses | Est. MY MOS | Pricing | Source/Notes |
|----------|-----------|------------|-------------|---------|--------------|
| **ElevenLabs** | Best voice cloning (3sec samples) | Premium price, no MY tuning | 4.0-4.3 | $0.30/1K chars | ⚠️ Estimated (excellent English, no MY test)¹ |
| **Google Cloud TTS** | Fast, WaveNet quality | Poor code-switching, no particles | 3.2-3.5 | $0.016/1K chars | ⚠️ Estimated from multilingual quality² |
| **AWS Polly** | Good infrastructure, Neural voices | Wrong accent, no MY support | 3.0-3.3 | $0.016/1K chars | ⚠️ Estimated (Standard voices ~2.5-3.0)³ |
| **Azure Speech** | Enterprise features, Custom voices | Expensive custom training | 3.3-3.6 | $0.016/1K chars | ⚠️ Estimated from Neural voices⁴ |
| **PlayHT** | Good multilingual, affordable | Limited code-switching | 3.5-3.8 | $0.019/1K chars | ⚠️ Estimated from user reviews⁵ |
| **Resemble.ai** | Voice cloning, real-time | No Malaysian optimization | 3.6-3.9 | $0.006/sec (~$0.18/1K) | ⚠️ Estimated (good cloning quality)⁶ |
| **XTTS v2 (Coqui)** | Open-source, voice cloning | Requires fine-tuning | 3.4-3.7 | Free (self-host) | Baseline for our fine-tuning⁷ |
| **Our Solution** | Malaysian-optimized | New entrant, limited voices | **4.0-4.2** ✓ | **$0.15/1K chars** | 🎯 **Target** (requires validation) |

**MOS (Mean Opinion Score) Scale Reference:**
- **5.0**: Perfect (indistinguishable from human)
- **4.0-4.5**: Excellent (natural, minor imperfections)
- **3.5-4.0**: Good (acceptable for most use cases)
- **3.0-3.5**: Fair (robotic at times, usable)
- **< 3.0**: Poor (clearly synthetic)

**Sources & Methodology:**
1. **ElevenLabs**: Based on third-party reviews and benchmarks ([TTS Arena 2024](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)). Widely considered best-in-class for English (MOS ~4.3-4.6), but no Malaysian testing published. Code-switching likely degrades by 0.2-0.5 MOS.
2. **Google Cloud TTS**: WaveNet quality generally rates 3.5-4.0 MOS on single-language tasks ([Google Research, 2018](https://arxiv.org/abs/1609.03499)). Code-switching typically reduces by 0.3-0.5 MOS based on literature.
3. **AWS Polly**: Neural voices show ~3.5-3.8 MOS on English ([AWS Performance Metrics](https://aws.amazon.com/polly/)). Malaysian accent mismatch and no particle support likely reduce to 3.0-3.3.
4. **Azure Speech**: Neural voices perform at 3.8-4.0 MOS for single languages. Estimated degradation for Malaysian context.
5. **PlayHT**: Based on user reviews on G2/Capterra (avg 4.2/5.0 user rating, ~3.5-3.8 MOS equivalent).
6. **Resemble.ai**: Voice cloning quality generally high (3.8-4.0 MOS for cloned voices), but no Malaysian testing.
7. **XTTS v2**: Baseline quality 3.5-3.8 MOS on English. Fine-tuning on Malaysian data should improve to 4.0+ MOS target.

**Key Differentiators:**
1. ✅ **Malay-Primary Optimization**: Optimized for Malay (60-70%) with natural English code-switching (20-30%)
2. ✅ **Particle-Aware**: Proper pronunciation of "lah", "leh", "loh", "meh", "lor"
3. ✅ **Local Vocabulary**: Handles slang, Pinyin, and Malaysian-specific terms naturally
4. ✅ **Authentic Accent**: Trained on real Malaysian speakers (Malay and English accents)
5. ✅ **Cost-Effective**: $0.15/1K chars vs $0.30/1K for ElevenLabs (50% savings)
6. ✅ **Self-Hostable**: Open-source base (XTTS v2) allows on-premise deployment
7. ✅ **Complete Suite**: Paired with ASR for full voice AI solution

**Recommended Validation Approach:**
Before making public claims, we must:
1. ✅ **MOS Testing**: 20+ Malaysian raters, 100+ samples, standardized protocol
2. ✅ **A/B Comparison**: Test all competitors on **same text samples**
3. ✅ **Code-Switching Test**: Specific evaluation of mixed-language sentences
4. ✅ **Particle Evaluation**: Dedicated test for discourse marker pronunciation
5. ✅ **Publish Results**: Transparent methodology and raw data
6. ✅ **Update Table**: Replace estimates with actual measured MOS scores

**Competitive Analysis Details:**

**ElevenLabs** (Primary TTS Competitor)
- **Strengths:**
  - Industry-leading voice cloning (requires only 3-second sample)
  - Exceptional naturalness (MOS ~4.3-4.6 on English)
  - Real-time streaming API
  - 29+ languages supported
  - Strong brand and user base (1M+ users)
- **Weaknesses:**
  - Premium pricing ($0.30/1K chars, 2x our price)
  - No Malaysian-specific optimization
  - Limited code-switching support (treats as separate languages)
  - No discourse particle handling
  - Closed-source (no self-hosting option)
- **Our Advantage:** Malaysian-specific training at half the cost

**Google Cloud TTS** (Enterprise Standard)
- **Strengths:**
  - Fast inference (RTF ~0.15)
  - WaveNet quality (good for single languages)
  - Strong infrastructure and reliability
  - Reasonable pricing ($0.016/1K chars)
- **Weaknesses:**
  - Poor code-switching (treats as language errors)
  - No Malaysian accent option
  - Cannot handle particles naturally
  - Limited customization without expensive custom voice training ($10K+)
- **Our Advantage:** Native code-switching support, authentic Malaysian voice

**XTTS v2 (Open-Source Baseline)**
- **Strengths:**
  - Free and open-source
  - Voice cloning capability
  - Active community and development
  - Can be fine-tuned for specific use cases
- **Weaknesses:**
  - Requires significant ML expertise to deploy
  - Baseline quality insufficient for Malaysian (3.4-3.7 MOS)
  - No pre-trained Malaysian models
  - Self-hosting infrastructure costs
- **Our Advantage:** We provide fine-tuned, production-ready version with Malaysian optimization

---

## 2. Product Goals & Objectives

### 2.1 North Star Metrics

**Primary North Star:**
- **Monthly Audio Generation Hours**: 10,000 hours by Month 12

**Supporting Metrics:**
- Monthly Active Users (MAU): 5,000 by Month 12
- Daily Active Users (DAU): 1,500 by Month 12
- Net Promoter Score (NPS): >50

### 2.2 Product Objectives

**Quality Objectives:**
- Mean Opinion Score (MOS) > 4.0/5.0 for naturalness
- Code-switching accuracy > 95%
- Particle intonation quality > 4.2/5.0
- Word Error Rate (ASR round-trip) < 5%

**Performance Objectives:**
- API response time (p95) < 500ms
- Real-Time Factor (RTF) < 0.3
- System uptime > 99.5%
- Support 1000+ concurrent requests

**User Experience Objectives:**
- Time to first audio < 200ms (streaming)
- Setup/integration time < 30 minutes
- API success rate > 99%
- Documentation completeness score > 4.5/5

---

## 3. Functional Requirements

### 3.1 Core Features (MVP)

#### Feature 1: Multi-Language Text Input
**Description:** Accept text input in mixed Malay, English, and Pinyin

**Requirements:**
- Support UTF-8 text input
- Handle sentences with 1-3 languages mixed
- Accept input via REST API (JSON)
- Maximum input length: 5,000 characters per request
- Automatic language detection at word level
- Support for language hints/tags (optional)

**User Stories:**
- As a content creator, I want to input mixed-language text so that I can generate authentic Malaysian speech
- As a developer, I want automatic language detection so that I don't need to manually tag each word

**Acceptance Criteria:**
```
✓ System accepts mixed-language input without errors
✓ Language detection accuracy > 95% for common words
✓ API returns clear error messages for unsupported characters
✓ Response time < 100ms for text processing
```

#### Feature 2: Malaysian Particle Support
**Description:** Properly pronounce Malaysian discourse particles with contextual intonation

**Requirements:**
- Support particles: lah, leh, loh, lor, meh, mah, wat, ah, hor, sia
- Context-aware intonation (emphatic, questioning, casual)
- Natural pitch contours for each particle type
- Appropriate duration and stress patterns

**User Stories:**
- As a Malaysian user, I want particles to sound natural so that the speech feels authentic
- As a content creator, I want "lah" to sound different in "can lah" vs "like that lah"

**Acceptance Criteria:**
```
✓ All 10 common particles are recognized
✓ Native speaker MOS for particles > 4.2/5.0
✓ Correct rising/falling intonation based on context
✓ Natural integration with preceding word
```

#### Feature 3: Code-Switching Speech Generation
**Description:** Generate natural-sounding speech with smooth language transitions

**Requirements:**
- Seamless transitions between languages
- Maintain prosody continuity across boundaries
- Correct pronunciation in each language
- Natural rhythm for mixed sentences

**User Stories:**
- As a podcaster, I want smooth transitions between languages so that my audio sounds professional
- As an educator, I want each language to be clearly pronounced so students can learn properly

**Acceptance Criteria:**
```
✓ No audible "jumps" or discontinuities at language boundaries
✓ Code-switching accuracy > 95%
✓ Prosody naturalness score > 4.0/5.0
✓ Each language segment intelligible (WER < 5%)
```

#### Feature 4: Audio Output Generation
**Description:** Generate high-quality audio output in standard formats

**Requirements:**
- Output formats: WAV, MP3, OGG
- Sample rates: 16kHz, 22.05kHz, 24kHz, 44.1kHz
- Bit depths: 16-bit, 24-bit
- Stereo or mono output
- Adjustable quality settings

**User Stories:**
- As a developer, I want multiple format options so that I can integrate with my existing system
- As a content creator, I want high-quality audio so that my content sounds professional

**Acceptance Criteria:**
```
✓ All output formats produce valid audio files
✓ Audio quality (SNR) > 35 dB
✓ No artifacts (clicks, pops, distortion)
✓ Consistent loudness (LUFS -16 to -20)
```

#### Feature 5: REST API Access
**Description:** Provide RESTful API for programmatic access

**Requirements:**
- POST /synthesize endpoint
- Authentication via API key
- Request throttling and rate limiting
- Comprehensive error handling
- JSON request/response format

**User Stories:**
- As a developer, I want a simple REST API so that I can integrate TTS into my application
- As a product manager, I want rate limiting so that we can manage costs and prevent abuse

**Acceptance Criteria:**
```
✓ API responds within 500ms (p95)
✓ Clear API documentation with examples
✓ Proper HTTP status codes for all error cases
✓ Rate limits enforced (e.g., 100 requests/min)
```

### 3.2 Advanced Features (Post-MVP)

#### Feature 6: Voice Selection
**Requirements:**
- 3-5 voice options (male/female, different ages)
- Consistent quality across all voices
- Voice preview samples
- Speaker ID parameter in API

**Priority:** High  
**Timeline:** Month 6-8

#### Feature 7: Speed Control
**Requirements:**
- Speed range: 0.5x to 2.0x
- Maintain pitch during speed changes
- No quality degradation at 0.75x-1.25x range
- Speed parameter in API

**Priority:** High  
**Timeline:** Month 4-6

#### Feature 8: Streaming Output
**Requirements:**
- Chunk-based generation
- Start playback before full synthesis
- Latency < 200ms to first audio chunk
- WebSocket or HTTP streaming support

**Priority:** High  
**Timeline:** Month 8-10

#### Feature 9: Prosody Control (SSML)
**Requirements:**
- Support SSML tags: <break>, <emphasis>, <prosody>
- Control pitch, rate, volume
- Phoneme-level control
- Backward compatible with plain text

**Priority:** Medium  
**Timeline:** Month 10-12

#### Feature 10: Emotion/Style Control
**Requirements:**
- Emotion options: neutral, happy, sad, excited, professional
- Style options: formal, casual, energetic, calm
- Consistent quality across emotions
- Emotion parameter in API

**Priority:** Medium  
**Timeline:** Post-launch (Month 13+)

#### Feature 11: Voice Cloning
**Requirements:**
- Upload 1-5 minutes of target voice
- Generate speech in that voice style
- Quality comparable to base voices
- Privacy-preserving processing

**Priority:** Low  
**Timeline:** Post-launch (Month 15+)

### 3.3 Platform Features

#### Feature 12: Web Playground
**Requirements:**
- Browser-based testing interface
- Real-time synthesis preview
- Save and share generated audio
- Parameter adjustment UI
- No login required for basic testing

**Priority:** High  
**Timeline:** Month 3-4

#### Feature 13: API Dashboard
**Requirements:**
- Usage statistics and analytics
- API key management
- Billing and payment history
- Usage alerts and notifications
- Download invoices

**Priority:** High  
**Timeline:** Month 5-6

#### Feature 14: Documentation Portal
**Requirements:**
- Getting started guide
- API reference with examples
- Code samples (Python, JavaScript, cURL)
- Interactive API explorer
- FAQ and troubleshooting

**Priority:** High  
**Timeline:** Month 4-5

#### Feature 15: SDKs & Libraries
**Requirements:**
- Python SDK
- JavaScript/Node.js SDK
- Example applications
- Unit tests and type definitions

**Priority:** Medium  
**Timeline:** Month 7-9

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Latency (p50) | < 200ms | Time from request to first byte |
| API Latency (p95) | < 500ms | 95th percentile response time |
| API Latency (p99) | < 1000ms | 99th percentile response time |
| Real-Time Factor | < 0.3 | Synthesis time / audio duration |
| Throughput | 100+ req/s | Per server instance |
| Concurrent Users | 1000+ | Simultaneous active sessions |

### 4.2 Quality Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| MOS Naturalness | > 4.0/5.0 | Subjective evaluation (30+ raters) |
| MOS Prosody | > 4.0/5.0 | Rhythm and intonation quality |
| Particle Quality | > 4.2/5.0 | Particle-specific evaluation |
| Code-Switch Quality | > 4.0/5.0 | Transition smoothness |
| Intelligibility (WER) | < 5% | ASR round-trip test |
| Language Accuracy | > 95% | Correct pronunciation per language |

### 4.3 Reliability Requirements

| Requirement | Target | Notes |
|-------------|--------|-------|
| Uptime SLA | 99.5% | ~3.6 hours downtime/month |
| Error Rate | < 0.5% | Failed requests / total requests |
| Data Durability | 99.999% | For stored audio/settings |
| Backup Frequency | Daily | Incremental backups |
| Disaster Recovery | < 4 hours | RTO (Recovery Time Objective) |
| Data Recovery Point | < 1 hour | RPO (Recovery Point Objective) |

### 4.4 Scalability Requirements

**Horizontal Scaling:**
- Support auto-scaling based on load
- Scale from 1 to 20+ instances seamlessly
- Handle traffic spikes (5x normal load)

**Data Scaling:**
- Support 1M+ API requests per day
- Store 100,000+ user accounts
- Handle 10TB+ audio generation per month

**Geographic Scaling:**
- Deployable in multiple regions
- Edge caching for low latency
- CDN integration for audio delivery

### 4.5 Compatibility Requirements

**Client Compatibility:**
- All modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Chrome Mobile)
- HTTP/1.1 and HTTP/2
- IPv4 and IPv6

**Integration Compatibility:**
- RESTful API (language-agnostic)
- Python 3.8+
- Node.js 14+
- cURL and Postman

**Audio Format Compatibility:**
- WAV (PCM, ADPCM)
- MP3 (CBR, VBR)
- OGG Vorbis
- Future: AAC, FLAC

### 4.6 Operational Requirements

**Monitoring:**
- Real-time performance dashboard
- Error tracking and alerting
- Usage analytics
- Resource utilization metrics

**Logging:**
- Structured logging (JSON)
- Log retention: 90 days
- Searchable log aggregation
- PII filtering in logs

**Deployment:**
- Automated CI/CD pipeline
- Blue-green deployment
- Rollback capability < 5 minutes
- Canary releases for new features

**Maintenance:**
- Scheduled maintenance window: < 2 hours/month
- Zero-downtime deployments
- Automated health checks
- Self-healing infrastructure

---


## 5. User Interface Requirements

### 5.1 Web Playground

**Key Screens:**

1. **Main Synthesis Interface**
   - Text input area (5000 char limit)
   - Language tags (auto-detected, editable)
   - Voice selector dropdown
   - Speed slider (0.5x - 2.0x)
   - Generate button
   - Audio player with waveform visualization
   - Download button (WAV/MP3)

2. **Examples Gallery**
   - Pre-made examples by use case
   - Sample audio playback
   - Copy text to main input
   - Categories: Business, Education, Entertainment, Casual

3. **Settings Panel**
   - Audio format selection
   - Sample rate selection
   - Advanced parameters (for power users)

**UX Requirements:**
- Mobile-responsive design
- Keyboard shortcuts (Ctrl+Enter to generate)
- Drag-and-drop for text files
- Real-time character count
- Clear error messages with suggestions
- Loading states with progress indication

### 5.2 Developer Dashboard

**Key Screens:**

1. **Overview Dashboard**
   - Usage statistics (requests, characters, audio time)
   - Usage quotas and limits
   - Recent activity
   - Quick actions (generate API key, view docs)

2. **API Keys Management**
   - List all API keys
   - Create new key
   - Revoke key
   - Key usage statistics
   - Permissions management

3. **Usage Analytics**
   - Requests over time (chart)
   - Error rate trends
   - Popular endpoints
   - Peak usage times
   - Cost breakdown

4. **Account Management**
   - Account settings
   - Usage history
   - API quotas and limits
   - Usage forecasting
   - Export data

5. **Documentation**
   - Getting started guide
   - API reference
   - Code examples
   - Interactive API tester
   - Changelog

---

## 6. Competitive Analysis

### Competitive Landscape Overview

The TTS market has several players, but we have a unique positioning:

**Direct Competition:**
1. **ElevenLabs** - Most direct threat (voice cloning, high quality)
2. **Google Cloud TTS** - Enterprise, multilingual
3. **Amazon Polly** - Cloud integration
4. **Microsoft Azure Speech** - Enterprise features

**Our Differentiation:**
- 🇲🇾 **Only TTS optimized for Malaysian speech patterns**
- 🔀 **Only solution supporting Malay-English-Chinese code-switching**
- 🗣️ **Only TTS with authentic particle pronunciation**
- 💰 **Better value for Malaysian market (70-90% cheaper)**

**Key Insight:** No competitor offers Malaysian-specific features. They compete on quality, speed, and language count. We compete on cultural authenticity and local optimization.

---

### 6.1 Direct Competitors

#### Competitor 1: ElevenLabs ⚠️ **MOST DIRECT**

**Overview:** Leading AI voice cloning and TTS platform with focus on natural, expressive speech

**Strengths:**
- ✅ **Excellent Voice Cloning**: Industry-leading with 1-5 minutes of audio
- ✅ **High Quality**: Very natural-sounding voices (MOS ~4.3+)
- ✅ **Multilingual**: 29+ languages including English, Chinese
- ✅ **Expressive**: Good emotion and prosody control
- ✅ **Developer-Friendly**: Clean API, good documentation
- ✅ **Voice Library**: Large marketplace of pre-made voices
- ✅ **Fast Iteration**: Regular updates and improvements

**Weaknesses:**
- ❌ **No Malaysian-Specific Features**:
  - No Malaysian accent option
  - No code-switching support
  - No particle understanding (lah, leh, loh)
  - Standard Malay only (not Malaysian Malay)
- ❌ **Expensive**: 
  - Starter: $5/month (30k chars) → $167/1M chars
  - Creator: $22/month (100k chars) → $220/1M chars
  - Pro: $99/month (500k chars) → $198/1M chars
  - 10-50x more expensive than our target
- ❌ **No Pinyin Support**: Chinese is Mandarin only
- ❌ **Generic**: Not specialized for any regional variant
- ❌ **Occasional Artifacts**: Can have pronunciation issues
- ❌ **Limited Fine-tuning**: Voice cloning but no accent adaptation

**Our Advantage:**
- 🎯 **Malaysian-Specialized**: 
  - Authentic Malaysian accent (English + Malay)
  - Code-switching between 3 languages
  - Particle pronunciation with proper intonation
  - Malaysian Mandarin/Pinyin support
- 💰 **Better Value**: 70-90% cheaper for local market
- 🎨 **Purpose-Built**: Not generic, optimized for Malaysian speech patterns
- 🚀 **Local Focus**: Understand Malaysian market needs
- 📊 **Quality for Use Case**: Match or exceed for Malaysian content

**Market Positioning:**
- ElevenLabs: Global, premium, general-purpose
- Us: Regional, specialized, Malaysian-optimized

**Competitive Strategy:**
1. Don't compete on voice cloning technology (they're ahead)
2. Compete on Malaysian-specific features (we're unique)
3. Compete on value for Malaysian market (we're cheaper)
4. Partner potential: Could offer ElevenLabs as premium alternative

---

#### Competitor 2: Google Cloud Text-to-Speech
**Strengths:**
- High quality (WaveNet voices)
- Supports 40+ languages including Malay
- Strong brand recognition
- Reliable infrastructure

**Weaknesses:**
- No code-switching support
- Poor Malaysian accent (uses Standard Malay)
- No particle understanding
- Expensive ($4-$16 per 1M chars)
- Complex pricing

**Our Advantage:**
- Native Malaysian code-switching
- Authentic accent and particles
- 70% cheaper for Malaysian market
- Specialized, not general-purpose

#### Competitor 3: Amazon Polly
**Strengths:**
- AWS integration
- Multiple voice styles
- Neural voices available
- Good documentation

**Weaknesses:**
- Limited Malay support
- No Malaysian English
- No code-switching
- Pricing similar to Google

**Our Advantage:**
- Malaysian-specific features
- Better for local market
- Simpler pricing
- Focus on quality, not quantity of languages

#### Competitor 4: Microsoft Azure Speech
**Strengths:**
- Enterprise features
- Custom neural voice
- Good documentation
- SSML support

**Weaknesses:**
- No Malaysian-specific features
- Expensive custom voice ($2,500+)
- Complex setup
- Overkill for most use cases

**Our Advantage:**
- Specialized for Malaysian market
- Easier to use
- Better value for money
- Out-of-box code-switching

### 6.2 Indirect Competitors

#### Voice-Over Artists / Freelancers
**Strengths:**
- Human quality
- Flexible and creative
- Can do complex emotions

**Weaknesses:**
- Expensive (RM500-2000 per project)
- Slow turnaround (days)
- Not scalable
- Inconsistent quality

**Our Advantage:**
- Instant generation
- 95% cheaper
- Consistent quality
- Unlimited revisions

#### Local TTS Startups
**Strengths:**
- Local market knowledge
- Agile and focused

**Weaknesses:**
- Limited resources
- Unproven technology
- Small team

**Our Advantage:**
- Better technology (if we execute well)
- Faster go-to-market
- Strong technical team

---

## 7. Risks & Mitigation

### 7.1 Product Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Audio quality doesn't meet expectations | Medium | High | Rigorous testing, beta program, iterative improvement |
| Code-switching doesn't work well | Medium | High | Extensive training data, multi-task learning, expert evaluation |
| Particles sound unnatural | High | Medium | Dedicated particle dataset, native speaker feedback loop |
| Performance too slow for real-time | Low | Medium | Optimization from day 1, quantization, caching |
| Users don't value Malaysian-specific features | Low | High | User research upfront, beta testing, feedback collection |

### 7.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training data insufficient | Medium | High | Multiple data collection strategies, synthetic data, transfer learning |
| Model doesn't converge | Low | High | Proven architectures, pre-training, expert consultation |
| Infrastructure costs too high | Medium | Medium | Optimization, auto-scaling, cost monitoring, pricing adjustment |
| Security vulnerabilities | Low | High | Security audits, penetration testing, bug bounty program |

### 7.3 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Slow user adoption | Medium | High | Strong marketing, partnerships, free tier, referral program |
| Big tech (Google/Amazon) enters market | Low | High | Focus on quality and specialization, build community, fast iteration |
| Revenue doesn't cover costs | Medium | High | Lean operations, pricing optimization, enterprise sales focus |
| Key team members leave | Low | Medium | Documentation, knowledge sharing, competitive compensation |

---

## 8. Appendix

### 11.1 Glossary

- **Code-Switching:** Alternating between languages within a conversation or sentence
- **MOS (Mean Opinion Score):** Subjective quality rating from 1-5
- **Particle:** Discourse marker that adds emphasis or emotion (e.g., lah, leh)
- **RTF (Real-Time Factor):** Ratio of synthesis time to audio duration
- **TTS:** Text-to-Speech
- **WER (Word Error Rate):** Percentage of words incorrectly transcribed

### 11.2 References

- Malaysian Language Statistics: [Department of Statistics Malaysia]
- Voice Technology Market Research: [Grand View Research, 2024]
- TTS User Survey: [Internal Research, 2025]

### 11.3 Change Log

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Oct 12, 2025 | Initial draft | Product Team |

---

**Document Approval:**

- [ ] Product Manager
- [ ] Engineering Lead
- [ ] Design Lead
- [ ] Executive Sponsor

**Next Review Date:** November 12, 2025

