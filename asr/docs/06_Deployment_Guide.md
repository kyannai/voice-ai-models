# Deployment Guide
# Malaysian Multilingual ASR System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** DevOps & SRE Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-Deployment Checklist](#2-pre-deployment-checklist)
3. [Docker Containerization](#3-docker-containerization)
4. [Kubernetes Deployment](#4-kubernetes-deployment)
5. [API Gateway Setup](#5-api-gateway-setup)
6. [CI/CD Pipeline](#6-cicd-pipeline)
7. [Monitoring & Logging](#7-monitoring--logging)
8. [Security](#8-security)
9. [Scaling Strategy](#9-scaling-strategy)
10. [Disaster Recovery](#10-disaster-recovery)

---

## 1. Overview

### 1.1 Deployment Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      LOAD BALANCER                       │  │
│  │            (AWS ALB / GCP Load Balancer)                │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                        │                                       │
│  ┌─────────────────────▼────────────────────────────────────┐  │
│  │                   API GATEWAY                            │  │
│  │       (Kong / NGINX Ingress Controller)                 │  │
│  │   - Authentication (API Keys)                           │  │
│  │   - Rate Limiting                                        │  │
│  │   - TLS Termination                                      │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                        │                                       │
│        ┌───────────────┴──────────────┐                       │
│        │                              │                       │
│  ┌─────▼──────┐               ┌──────▼─────┐                 │
│  │ API Service│ (3+ replicas) │Worker Pool │ (2-10 replicas) │
│  │  (FastAPI) │               │  (Celery)  │                 │
│  │            │               │            │                 │
│  │ - Validate │               │ - Download │                 │
│  │ - Queue    │               │ - Process  │                 │
│  │ - Return   │               │ - Inference│                 │
│  └─────┬──────┘               └──────┬─────┘                 │
│        │                              │                       │
│  ┌─────▼──────────────────────────────▼─────┐                 │
│  │         Redis (Job Queue + Cache)        │                 │
│  └──────────────────────────────────────────┘                 │
│                        │                                       │
│  ┌─────────────────────▼────────────────────────────────────┐  │
│  │              ML INFERENCE LAYER                          │  │
│  │   - Whisper-large v3 (fine-tuned)                       │  │
│  │   - GPU: NVIDIA T4 / A10                                │  │
│  │   - Batch inference optimization                        │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                        │                                       │
│  ┌─────────────────────▼────────────────────────────────────┐  │
│  │              STORAGE & DATABASE                          │  │
│  │   - S3/GCS: Audio files, transcripts                    │  │
│  │   - PostgreSQL: Metadata, users                         │  │
│  │   - Redis: Cache, sessions                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            MONITORING & OBSERVABILITY                    │  │
│  │   - Prometheus: Metrics                                  │  │
│  │   - Grafana: Dashboards                                  │  │
│  │   - ELK Stack: Logs                                      │  │
│  │   - Sentry: Error tracking                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Infrastructure Requirements

**Production Environment:**

| Component | Specification | Quantity | Cost (AWS/month) |
|-----------|---------------|----------|------------------|
| **API Server** | t3.large (2 vCPU, 8GB RAM) | 3 | $150 |
| **Worker (GPU)** | g4dn.xlarge (T4 16GB) | 2-5 | $1,140-$2,850 |
| **Redis** | r6g.large (2 vCPU, 16GB) | 1 | $125 |
| **PostgreSQL** | db.t3.medium (2 vCPU, 4GB) | 1 | $60 |
| **S3 Storage** | Standard (1TB) | - | $23 |
| **Load Balancer** | ALB | 1 | $25 |
| **Total** | - | - | **$1,523-$3,233** |

**Staging Environment:** ~30% of production cost ($450-$970/month)

---

## 2. Pre-Deployment Checklist

### 2.1 Model Preparation

- [ ] Model trained and evaluated (WER < 15%)
- [ ] Model exported and tested
  ```bash
  python export_model.py --input ./whisper-malaysian-finetuned --output ./model_export/
  ```
- [ ] Inference benchmarked (RTF < 0.3)
- [ ] Model size optimized (< 5GB)

### 2.2 Code & Configuration

- [ ] API code reviewed and tested
- [ ] Environment variables documented
- [ ] Database migrations prepared
- [ ] API documentation (Swagger) generated

### 2.3 Infrastructure

- [ ] Cloud account set up (AWS/GCP/Azure)
- [ ] Kubernetes cluster provisioned
- [ ] GPU nodes available
- [ ] Domain name registered (api.asr.example.com)
- [ ] SSL certificate obtained (Let's Encrypt)

### 2.4 Security

- [ ] API key generation system implemented
- [ ] Rate limiting configured
- [ ] Data encryption enabled (TLS, AES-256)
- [ ] Secrets management set up (AWS Secrets Manager)
- [ ] PDPA compliance reviewed

### 2.5 Monitoring

- [ ] Prometheus installed
- [ ] Grafana dashboards created
- [ ] Alert rules configured
- [ ] Log aggregation set up (ELK/CloudWatch)
- [ ] Error tracking enabled (Sentry)

---

## 3. Docker Containerization

### 3.1 API Service Dockerfile

```dockerfile
# Dockerfile.api
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY ./config ./config

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 3.2 Worker Service Dockerfile

```dockerfile
# Dockerfile.worker
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.worker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.worker.txt

# Copy application code
COPY ./app ./app
COPY ./config ./config
COPY ./models ./models  # Pre-trained model

# Run Celery worker
CMD ["celery", "-A", "app.worker", "worker", "--loglevel=info", "--concurrency=2"]
```

### 3.3 Requirements Files

**requirements.txt (API Service):**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
celery==5.3.4
redis==5.0.1
python-multipart==0.0.6
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
boto3==1.29.7  # For S3
python-jose[cryptography]==3.3.0  # For JWT
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
httpx==0.25.2
```

**requirements.worker.txt (Worker Service):**
```txt
# Include API requirements
-r requirements.txt

# ML dependencies
torch==2.1.0
torchaudio==2.1.0
transformers==4.35.0
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.3
jiwer==3.0.3

# Unsloth (for inference with LoRA)
unsloth @ git+https://github.com/unslothai/unsloth.git
```

### 3.4 Build and Push

```bash
# Build images
docker build -f Dockerfile.api -t asr-api:v1.0 .
docker build -f Dockerfile.worker -t asr-worker:v1.0 .

# Tag for registry
docker tag asr-api:v1.0 yourusername/asr-api:v1.0
docker tag asr-worker:v1.0 yourusername/asr-worker:v1.0

# Push to Docker Hub / ECR / GCR
docker push yourusername/asr-api:v1.0
docker push yourusername/asr-worker:v1.0
```

---

## 4. Kubernetes Deployment

### 4.1 Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: asr-system
```

### 4.2 ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: asr-config
  namespace: asr-system
data:
  REDIS_URL: "redis://redis-service:6379/0"
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "asr_db"
  MODEL_PATH: "/models/whisper-malaysian"
  LOG_LEVEL: "INFO"
  MAX_AUDIO_SIZE_MB: "500"
  MAX_AUDIO_DURATION_SEC: "10800"  # 3 hours
```

### 4.3 Secrets

```bash
# Create secrets (do not commit to git!)
kubectl create secret generic asr-secrets \
  --from-literal=POSTGRES_USER=asr_user \
  --from-literal=POSTGRES_PASSWORD=your_secure_password \
  --from-literal=JWT_SECRET_KEY=your_jwt_secret \
  --from-literal=AWS_ACCESS_KEY_ID=your_aws_key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your_aws_secret \
  -n asr-system
```

### 4.4 API Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-api
  namespace: asr-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: asr-api
  template:
    metadata:
      labels:
        app: asr-api
    spec:
      containers:
      - name: api
        image: yourusername/asr-api:v1.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: asr-config
        - secretRef:
            name: asr-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5

---
# api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: asr-api-service
  namespace: asr-system
spec:
  selector:
    app: asr-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### 4.5 Worker Deployment (GPU)

```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-worker
  namespace: asr-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: asr-worker
  template:
    metadata:
      labels:
        app: asr-worker
    spec:
      # GPU node selector
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      
      containers:
      - name: worker
        image: yourusername/asr-worker:v1.0
        envFrom:
        - configMapRef:
            name: asr-config
        - secretRef:
            name: asr-secrets
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
            memory: "16Gi"
            cpu: "4000m"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### 4.6 Redis

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: asr-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: asr-system
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
```

### 4.7 PostgreSQL

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: asr-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        envFrom:
        - configMapRef:
            name: asr-config
        - secretRef:
            name: asr-secrets
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: asr-system
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
```

### 4.8 Persistent Volume Claims

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: asr-system
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: asr-system
spec:
  accessModes:
  - ReadOnlyMany  # Multiple workers can read
  resources:
    requests:
      storage: 20Gi
```

### 4.9 Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: asr-api-hpa
  namespace: asr-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: asr-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: asr-worker-hpa
  namespace: asr-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: asr-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

### 4.10 Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f worker-deployment.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n asr-system
kubectl get services -n asr-system

# View logs
kubectl logs -f deployment/asr-api -n asr-system
kubectl logs -f deployment/asr-worker -n asr-system
```

---

## 5. API Gateway Setup

### 5.1 Ingress (NGINX)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: asr-ingress
  namespace: asr-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"  # 100 req/min per IP
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.asr.example.com
    secretName: asr-tls-secret
  rules:
  - host: api.asr.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: asr-api-service
            port:
              number: 80
```

### 5.2 Rate Limiting (Application Level)

```python
# app/middleware/rate_limiter.py
from fastapi import Request, HTTPException
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

# Initialize rate limiter
async def init_rate_limiter():
    redis_client = await redis.from_url("redis://redis-service:6379", encoding="utf-8")
    await FastAPILimiter.init(redis_client)

# Apply to endpoints
@app.post("/v1/transcribe")
@limiter.limit("100/hour")  # 100 requests per hour per API key
async def transcribe(request: Request, api_key: str = Depends(verify_api_key)):
    # ...
    pass
```

---

## 6. CI/CD Pipeline

### 6.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy ASR Service

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  DOCKER_REGISTRY: docker.io
  IMAGE_NAME_API: yourusername/asr-api
  IMAGE_NAME_WORKER: yourusername/asr-worker

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push API image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.api
          push: true
          tags: |
            ${{ env.IMAGE_NAME_API }}:latest
            ${{ env.IMAGE_NAME_API }}:${{ github.sha }}
      
      - name: Build and push Worker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.worker
          push: true
          tags: |
            ${{ env.IMAGE_NAME_WORKER }}:latest
            ${{ env.IMAGE_NAME_WORKER }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.27.0'
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
      
      - name: Update deployments
        run: |
          kubectl set image deployment/asr-api \
            api=${{ env.IMAGE_NAME_API }}:${{ github.sha }} \
            -n asr-system
          
          kubectl set image deployment/asr-worker \
            worker=${{ env.IMAGE_NAME_WORKER }}:${{ github.sha }} \
            -n asr-system
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/asr-api -n asr-system --timeout=5m
          kubectl rollout status deployment/asr-worker -n asr-system --timeout=5m
      
      - name: Run smoke tests
        run: |
          ./scripts/smoke_test.sh api.asr.example.com
```

---

## 7. Monitoring & Logging

### 7.1 Prometheus Metrics

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
asr_requests_total = Counter(
    "asr_requests_total",
    "Total ASR requests",
    ["method", "endpoint", "status"],
)

# Processing time
asr_processing_duration = Histogram(
    "asr_processing_duration_seconds",
    "Time to process transcription",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Real-Time Factor
asr_rtf = Histogram(
    "asr_rtf",
    "Real-Time Factor (processing_time / audio_duration)",
    buckets=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0],
)

# WER estimate (from sampled user feedback)
asr_wer_estimate = Gauge(
    "asr_wer_estimate",
    "Estimated WER from production samples",
)

# Queue depth
celery_queue_length = Gauge(
    "celery_queue_length",
    "Number of tasks in Celery queue",
)
```

### 7.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Malaysian ASR System",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(asr_requests_total[5m])"
          }
        ]
      },
      {
        "title": "P50/P95/P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(asr_processing_duration_seconds_bucket[5m]))"
          },
          {
            "expr": "histogram_quantile(0.95, rate(asr_processing_duration_seconds_bucket[5m]))"
          },
          {
            "expr": "histogram_quantile(0.99, rate(asr_processing_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Real-Time Factor",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(asr_rtf_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(asr_requests_total{status=\"error\"}[5m])"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_duty_cycle"
          }
        ]
      }
    ]
  }
}
```

### 7.3 Alerting Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: asr_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(asr_requests_total{status="error"}[5m]) > 0.05
    for: 10m
    annotations:
      summary: "High error rate (> 5%)"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(asr_processing_duration_seconds_bucket[5m])) > 5
    for: 15m
    annotations:
      summary: "P95 latency > 5 seconds"
  
  - alert: LowGPUUtilization
    expr: nvidia_gpu_duty_cycle < 30
    for: 30m
    annotations:
      summary: "GPU underutilized (< 30%)"
  
  - alert: HighQueueDepth
    expr: celery_queue_length > 100
    for: 10m
    annotations:
      summary: "Task queue backing up (> 100 tasks)"
```

---

## 8. Security

### 8.1 API Key Management

```python
# app/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import hashlib

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    """Verify API key from Authorization header."""
    
    if not api_key_header or not api_key_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
    
    api_key = api_key_header.replace("Bearer ", "")
    
    # Hash and look up in database
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    user = await get_user_by_api_key_hash(key_hash)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    return user
```

### 8.2 Data Encryption

**At Rest (S3):**
```python
import boto3

s3_client = boto3.client('s3')

# Upload with server-side encryption
s3_client.put_object(
    Bucket='asr-audio-bucket',
    Key='audio/file.wav',
    Body=audio_data,
    ServerSideEncryption='AES256',  # or 'aws:kms'
)
```

**At Transit (TLS):**
```yaml
# Enforce HTTPS only
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
```

### 8.3 Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: asr-api-policy
  namespace: asr-system
spec:
  podSelector:
    matchLabels:
      app: asr-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

---

## 9. Scaling Strategy

### 9.1 Horizontal Scaling

**Auto-scaling based on metrics:**
- CPU > 70% → scale up API pods
- Queue depth > 10 → scale up worker pods
- Queue depth < 3 → scale down workers

### 9.2 Vertical Scaling

**When to scale vertically:**
- Consistent OOM errors → increase memory limits
- CPU throttling → increase CPU limits

### 9.3 Cost Optimization

**Spot Instances for Workers:**
```yaml
# Use spot instances (60-90% cheaper)
nodeSelector:
  eks.amazonaws.com/capacityType: SPOT
tolerations:
- key: "spot"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

**Cluster Autoscaler:**
```bash
# Install cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
```

---

## 10. Disaster Recovery

### 10.1 Backup Strategy

**Database Backups:**
```bash
# Automated PostgreSQL backups to S3 (daily)
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \  # 2 AM daily
  -- /bin/sh -c "pg_dump -h postgres-service -U asr_user asr_db | gzip | aws s3 cp - s3://asr-backups/postgres/backup-$(date +%Y%m%d).sql.gz"
```

**Model Backups:**
- Store models in S3 with versioning enabled
- Tag each model with training date and metrics

### 10.2 Disaster Recovery Plan

**RPO (Recovery Point Objective):** 24 hours (daily backups)  
**RTO (Recovery Time Objective):** 2 hours

**Recovery Procedure:**
1. Provision new Kubernetes cluster (if needed)
2. Restore database from latest backup
3. Deploy application from Docker images
4. Load model from S3
5. Update DNS to point to new cluster
6. Run smoke tests

---

## 11. Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (unit, integration, E2E)
- [ ] Model performance validated (WER < 15%)
- [ ] Security review completed
- [ ] Load testing completed (1000 concurrent requests)
- [ ] Documentation updated

### Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests on staging
- [ ] Deploy to production (blue-green or canary)
- [ ] Monitor metrics for 1 hour
- [ ] Notify stakeholders

### Post-Deployment
- [ ] Verify all services healthy
- [ ] Check error rates and latency
- [ ] Review first 100 production transcriptions
- [ ] Update status page
- [ ] Document any issues

---

**End of Deployment Guide**

*For project timeline and execution, see Project Execution Plan (07).*

