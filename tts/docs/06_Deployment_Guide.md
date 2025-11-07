# Deployment Guide
# Malaysian Multilingual TTS System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** DevOps & Engineering Team

---

## 1. Deployment Overview

### 1.1 Deployment Strategy

We will use a **phased deployment** approach:

```
Phase 1: Development Environment
    ├─ Local development and testing
    └─ Duration: Throughout development

Phase 2: Staging Environment
    ├─ Pre-production testing
    ├─ Load testing and optimization
    └─ Duration: 2-3 weeks before launch

Phase 3: Production (Canary)
    ├─ 5% traffic to new version
    ├─ Monitor metrics closely
    └─ Duration: 3-7 days

Phase 4: Production (Full Release)
    ├─ 100% traffic
    ├─ Continuous monitoring
    └─ Rollback plan ready
```

### 1.2 Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CDN (CloudFlare / AWS CloudFront)       │
│                  • Static assets caching                 │
│                  • DDoS protection                       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Load Balancer (AWS ALB / NGINX)         │
│                  • SSL termination                       │
│                  • Health checks                         │
│                  • Request routing                       │
└───────┬──────────────────────────┬──────────────────────┘
        │                          │
┌───────▼──────────┐      ┌───────▼──────────┐
│  API Gateway     │      │  Web Dashboard   │
│  (Auto-scaled)   │      │  (Static S3)     │
│  • Auth          │      │                  │
│  • Rate limiting │      │                  │
└────────┬─────────┘      └──────────────────┘
         │
    ┌────▼─────┬─────────┬──────────┐
    │          │         │          │
┌───▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐
│ TTS   │  │ TTS  │  │ TTS  │  │ TTS  │
│Server1│  │Server│  │Server│  │Server│
│(GPU)  │  │  2   │  │  3   │  │  N   │
└───┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
    │         │         │         │
    └─────────┴─────────┴─────────┘
              │
    ┌─────────▼──────────┐
    │   Redis Cache      │
    │   • Results cache  │
    │   • Rate limiting  │
    └────────────────────┘
              │
    ┌─────────▼──────────┐
    │   PostgreSQL       │
    │   • User data      │
    │   • Usage logs     │
    └────────────────────┘
              │
    ┌─────────▼──────────┐
    │   S3 / MinIO       │
    │   • Generated audio│
    │   • Model files    │
    └────────────────────┘
```

---

## 2. Infrastructure Setup

### 2.1 Cloud Provider Selection

#### Option 1: AWS (Recommended)

**Pros:**
- Mature ML services (SageMaker, EC2 with GPUs)
- Global infrastructure
- Comprehensive monitoring tools
- Enterprise support

**Cons:**
- More expensive
- Complex pricing

**Estimated Monthly Cost:**
- EC2 p3.2xlarge (1 GPU): $3.06/hour × 730 hours = $2,234
- Load Balancer: $22
- RDS PostgreSQL (db.t3.medium): $50
- ElastiCache Redis: $30
- S3 Storage (1TB): $23
- Data Transfer: $50-100
- **Total: ~$2,400-2,500/month**

#### Option 2: GCP

**Pros:**
- Good AI/ML tools
- Competitive GPU pricing
- Clean interface

**Estimated Monthly Cost:**
- Similar to AWS: $2,300-2,400/month

#### Option 3: On-Premise

**Pros:**
- One-time cost
- Full control
- Data stays on-premise

**Cons:**
- High upfront investment
- Maintenance overhead
- No auto-scaling

**Initial Investment:**
- 4× RTX 3090 GPUs: $8,000-10,000
- Server hardware: $5,000-8,000
- Networking: $2,000-3,000
- **Total: ~$15,000-21,000**

**Operating Cost:**
- Electricity: $200-300/month
- Internet: $100-200/month
- Maintenance: $500-1,000/month

**Recommendation**: Start with AWS for flexibility, consider on-premise for scale (Year 2+)

### 2.2 Infrastructure as Code (Terraform)

```hcl
# terraform/main.tf

provider "aws" {
  region = "ap-southeast-1"  # Singapore
}

# VPC
resource "aws_vpc" "tts_vpc" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "malaysian-tts-vpc"
  }
}

# Subnets
resource "aws_subnet" "public_subnet" {
  count             = 2
  vpc_id            = aws_vpc.tts_vpc.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "malaysian-tts-public-${count.index}"
  }
}

# Security Group
resource "aws_security_group" "tts_sg" {
  name        = "malaysian-tts-sg"
  description = "Security group for TTS servers"
  vpc_id      = aws_vpc.tts_vpc.id
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["YOUR_IP/32"]  # SSH access
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instances (TTS Servers)
resource "aws_instance" "tts_server" {
  count         = 3
  ami           = "ami-gpu-optimized"  # Deep Learning AMI
  instance_type = "p3.2xlarge"  # 1x V100 GPU
  
  subnet_id              = aws_subnet.public_subnet[count.index % 2].id
  vpc_security_group_ids = [aws_security_group.tts_sg.id]
  
  key_name = aws_key_pair.tts_keypair.key_name
  
  root_block_device {
    volume_size = 200  # GB
    volume_type = "gp3"
  }
  
  user_data = file("${path.module}/scripts/setup_tts_server.sh")
  
  tags = {
    Name = "malaysian-tts-server-${count.index}"
  }
}

# Application Load Balancer
resource "aws_lb" "tts_alb" {
  name               = "malaysian-tts-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.tts_sg.id]
  subnets            = aws_subnet.public_subnet[*].id
  
  enable_deletion_protection = true
  
  tags = {
    Name = "malaysian-tts-alb"
  }
}

# Target Group
resource "aws_lb_target_group" "tts_tg" {
  name     = "malaysian-tts-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.tts_vpc.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "tts_db" {
  identifier           = "malaysian-tts-db"
  engine              = "postgres"
  engine_version      = "15.3"
  instance_class      = "db.t3.medium"
  allocated_storage   = 100
  storage_type        = "gp3"
  
  db_name  = "ttsdb"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.tts_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.tts_db_subnet.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "malaysian-tts-db-final"
  
  tags = {
    Name = "malaysian-tts-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "tts_redis" {
  cluster_id           = "malaysian-tts-redis"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  
  subnet_group_name  = aws_elasticache_subnet_group.tts_redis_subnet.name
  security_group_ids = [aws_security_group.tts_sg.id]
  
  tags = {
    Name = "malaysian-tts-redis"
  }
}

# S3 Bucket for audio storage
resource "aws_s3_bucket" "tts_audio" {
  bucket = "malaysian-tts-audio-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name = "malaysian-tts-audio"
  }
}

resource "aws_s3_bucket_versioning" "tts_audio_versioning" {
  bucket = aws_s3_bucket.tts_audio.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "tts_logs" {
  name              = "/aws/tts/malaysian-tts"
  retention_in_days = 30
  
  tags = {
    Name = "malaysian-tts-logs"
  }
}
```

---

## 3. Docker Containerization

### 3.1 Dockerfile

```dockerfile
# Dockerfile

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kuala_Lumpur

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/logs /app/cache

# Download models (or mount as volume)
ARG MODEL_VERSION=v1.0
RUN python3 scripts/download_models.py --version ${MODEL_VERSION}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 3.2 Docker Compose (Development)

```yaml
# docker-compose.yml

version: '3.8'

services:
  tts-api:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        MODEL_VERSION: v1.0
    image: malaysian-tts:latest
    container_name: tts-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://ttsuser:ttspass@postgres:5432/ttsdb
      - S3_BUCKET=local-audio-storage
      - S3_ENDPOINT=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres
      - minio
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: tts-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: tts-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=ttsuser
      - POSTGRES_PASSWORD=ttspass
      - POSTGRES_DB=ttsdb
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    container_name: tts-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio-data:/data
    command: server /data --console-address ":9001"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: tts-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: tts-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  minio-data:
  prometheus-data:
  grafana-data:
```

### 3.3 Build and Run

```bash
# Build image
docker build -t malaysian-tts:v1.0 .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f tts-api

# Stop
docker-compose down

# Clean up
docker-compose down -v  # Also removes volumes
```

---

## 4. Kubernetes Deployment

### 4.1 Kubernetes Manifests

#### Deployment

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: malaysian-tts
  namespace: tts-production
  labels:
    app: malaysian-tts
    version: v1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: malaysian-tts
  template:
    metadata:
      labels:
        app: malaysian-tts
        version: v1.0
    spec:
      containers:
      - name: tts-api
        image: youracr.azurecr.io/malaysian-tts:v1.0
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: "1"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: database-url
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: aws-access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: aws-secret-key
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: cache-storage
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: cache-storage
        emptyDir:
          sizeLimit: 10Gi
      nodeSelector:
        node.kubernetes.io/instance-type: p3.2xlarge  # GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

#### Service

```yaml
# k8s/service.yaml

apiVersion: v1
kind: Service
metadata:
  name: tts-service
  namespace: tts-production
  labels:
    app: malaysian-tts
spec:
  type: LoadBalancer
  selector:
    app: malaysian-tts
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

#### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tts-hpa
  namespace: tts-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: malaysian-tts
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### 4.2 Deployment Commands

```bash
# Create namespace
kubectl create namespace tts-production

# Create secrets
kubectl create secret generic tts-secrets \
  --from-literal=redis-url='redis://...' \
  --from-literal=database-url='postgresql://...' \
  --from-literal=aws-access-key='...' \
  --from-literal=aws-secret-key='...' \
  -n tts-production

# Apply manifests
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods -n tts-production
kubectl get svc -n tts-production

# View logs
kubectl logs -f deployment/malaysian-tts -n tts-production

# Scale manually
kubectl scale deployment malaysian-tts --replicas=5 -n tts-production
```

---

## 5. CI/CD Pipeline

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml

name: Build and Deploy

on:
  push:
    branches:
      - main
      - staging
  pull_request:
    branches:
      - main

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

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
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/staging'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-southeast-1
      
      - name: Deploy to ECS (Staging)
        run: |
          aws ecs update-service \
            --cluster tts-staging-cluster \
            --service tts-staging-service \
            --force-new-deployment

  deploy-production:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-southeast-1
      
      - name: Update Kubernetes deployment
        run: |
          kubectl set image deployment/malaysian-tts \
            tts-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n tts-production
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/malaysian-tts -n tts-production
      
      - name: Notify Slack
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "✅ Deployment successful: ${{ github.sha }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 6. Monitoring & Observability

### 6.1 Prometheus Configuration

```yaml
# monitoring/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'tts-api'
    static_configs:
      - targets: ['tts-api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### 6.2 Grafana Dashboards

Create dashboards for:

1. **API Performance**
   - Request rate
   - Latency (p50, p95, p99)
   - Error rate
   - RTF distribution

2. **System Health**
   - CPU usage
   - Memory usage
   - GPU utilization
   - Disk I/O

3. **Business Metrics**
   - Daily active users
   - Audio generation volume
   - Popular languages/features
   - User satisfaction (if feedback collected)

### 6.3 Alerting Rules

```yaml
# monitoring/alerting_rules.yml

groups:
  - name: tts_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(tts_latency_ms_bucket[5m])) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "p95 latency is {{ $value }}ms (threshold: 1000ms)"
      
      - alert: HighErrorRate
        expr: rate(tts_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 5%)"
      
      - alert: GPUMemoryHigh
        expr: gpu_memory_used_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU memory at {{ $value }}%"
      
      - alert: ServiceDown
        expr: up{job="tts-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TTS service is down"
          description: "Service {{ $labels.instance }} is not responding"
```

---

## 7. Security

### 7.1 API Security

```python
# api/security.py

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
import secrets
import hashlib
from datetime import datetime, timedelta

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class APIKeyManager:
    def __init__(self, db):
        self.db = db
    
    def generate_api_key(self, user_id):
        """Generate new API key"""
        key = f"tts_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Store hash in database
        self.db.insert_api_key(user_id, key_hash, datetime.now())
        
        return key  # Return only once!
    
    def validate_api_key(self, api_key: str):
        """Validate API key"""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check in database
        key_record = self.db.get_api_key(key_hash)
        
        if not key_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        if key_record['revoked']:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has been revoked"
            )
        
        # Update last used
        self.db.update_api_key_last_used(key_hash, datetime.now())
        
        return key_record

# Dependency
async def get_api_key(api_key_header: str = Security(api_key_header)):
    api_key_manager = APIKeyManager(db)
    return api_key_manager.validate_api_key(api_key_header)
```

### 7.2 Rate Limiting

```python
# api/rate_limiting.py

from fastapi import Request, HTTPException
from redis import Redis
import time

class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 60):
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User identifier
            limit: Max requests per window
            window: Time window in seconds
        """
        key = f"rate_limit:{user_id}"
        current_time = int(time.time())
        
        # Use sorted set to track requests
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, current_time - window)
        
        # Count requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {current_time: current_time})
        
        # Set expiry
        pipe.expire(key, window)
        
        results = pipe.execute()
        request_count = results[1]
        
        if request_count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window} seconds."
            )
        
        return True

# Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        user_id = request.state.user_id  # From API key validation
        tier = request.state.user_tier  # free, starter, pro, etc.
        
        # Different limits per tier
        limits = {
            'free': 10,
            'starter': 100,
            'pro': 500,
            'business': 2000
        }
        
        rate_limiter.check_rate_limit(user_id, limits.get(tier, 10))
    
    response = await call_next(request)
    return response
```

### 7.3 SSL/TLS Configuration

```nginx
# nginx/tts.conf

server {
    listen 80;
    server_name api.malaysian-tts.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.malaysian-tts.com;
    
    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/api.malaysian-tts.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.malaysian-tts.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
    
    location / {
        proxy_pass http://tts-backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://tts-backend/health;
        access_log off;
    }
}
```

---

## 8. Backup & Disaster Recovery

### 8.1 Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

# Database backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | \
  gzip > /backups/db_$(date +%Y%m%d_%H%M%S).sql.gz

# Upload to S3
aws s3 cp /backups/db_*.sql.gz s3://malaysian-tts-backups/database/

# Retain last 30 days
find /backups -name "db_*.sql.gz" -mtime +30 -delete

# Model checkpoints backup (weekly)
if [ $(date +%u) -eq 7 ]; then
  tar -czf /backups/models_$(date +%Y%m%d).tar.gz /app/models/
  aws s3 cp /backups/models_*.tar.gz s3://malaysian-tts-backups/models/
fi
```

### 8.2 Disaster Recovery Plan

**RTO (Recovery Time Objective):** 4 hours  
**RPO (Recovery Point Objective):** 1 hour

**Recovery Steps:**

1. **Database Failure**
   ```bash
   # Restore from latest backup
   aws s3 cp s3://malaysian-tts-backups/database/latest.sql.gz .
   gunzip latest.sql.gz
   psql -h $NEW_DB_HOST -U $DB_USER -d $DB_NAME < latest.sql
   
   # Update application config
   kubectl set env deployment/malaysian-tts \
     DATABASE_URL=postgresql://$NEW_DB_HOST/...
   ```

2. **Complete Region Failure**
   - Switch DNS to backup region
   - Deploy from Docker images
   - Restore data from S3 cross-region replication

3. **Model Corruption**
   - Download previous version from S3
   - Rollback Kubernetes deployment

---

## 9. Production Checklist

### 9.1 Pre-Launch Checklist

- [ ] **Infrastructure**
  - [ ] All services deployed and healthy
  - [ ] Load balancer configured
  - [ ] Auto-scaling rules in place
  - [ ] SSL certificates installed and valid

- [ ] **Security**
  - [ ] API authentication working
  - [ ] Rate limiting configured
  - [ ] Secrets properly managed
  - [ ] Security headers configured
  - [ ] Firewall rules configured

- [ ] **Monitoring**
  - [ ] Prometheus scraping metrics
  - [ ] Grafana dashboards created
  - [ ] Alerts configured
  - [ ] On-call rotation set up
  - [ ] Logging aggregation working

- [ ] **Data**
  - [ ] Database backed up
  - [ ] Models uploaded and accessible
  - [ ] Cache pre-warmed (if applicable)

- [ ] **Testing**
  - [ ] Load testing completed
  - [ ] API endpoints tested
  - [ ] Error handling verified
  - [ ] Rollback procedure tested

- [ ] **Documentation**
  - [ ] API documentation published
  - [ ] Runbooks created
  - [ ] Architecture diagrams updated
  - [ ] Contact information current

- [ ] **Legal & Compliance**
  - [ ] Terms of Service published
  - [ ] Privacy Policy published
  - [ ] GDPR compliance verified
  - [ ] Data retention policy documented

### 9.2 Post-Launch Checklist

- [ ] Monitor metrics for 48 hours
- [ ] Check error logs daily
- [ ] Verify billing/costs
- [ ] Collect user feedback
- [ ] Conduct postmortem meeting
- [ ] Document lessons learned

---

## 10. Troubleshooting

### 10.1 Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| High latency | p95 > 1s | Check GPU utilization; scale up if needed |
| OOM errors | Pods restarting | Increase memory limits; reduce batch size |
| Model not loading | Service won't start | Verify model files present; check permissions |
| Database connection | 502 errors | Check DB credentials; verify security groups |
| Rate limit errors | 429 responses | Adjust rate limits; check Redis connection |

### 10.2 Debugging Commands

```bash
# Check pod status
kubectl get pods -n tts-production

# View logs
kubectl logs -f deployment/malaysian-tts -n tts-production

# Execute into pod
kubectl exec -it malaysian-tts-xxx -n tts-production -- /bin/bash

# Check resource usage
kubectl top pods -n tts-production

# Describe pod (for events)
kubectl describe pod malaysian-tts-xxx -n tts-production

# Test API directly
curl -X POST https://api.malaysian-tts.com/v1/synthesize \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

---

**Document Version:** 1.0  
**Last Updated:** October 12, 2025  
**Next Review:** Before Production Launch

