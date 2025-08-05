# Multi-stage Dockerfile for ConfoRL production deployment
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.ai>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="ConfoRL" \
      org.label-schema.description="Adaptive Conformal Risk Control for Reinforcement Learning" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/danieleschmidt/conforl-lab" \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
COPY setup.py ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY conforl/ ./conforl/
COPY tests/ ./tests/

# Install the package
RUN pip install -e .

# Run tests during build (optional, can be disabled for faster builds)
ARG RUN_TESTS=true
RUN if [ "$RUN_TESTS" = "true" ]; then python -m pytest tests/ -x -v; fi

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r conforl && \
    useradd -r -g conforl -d /app -s /sbin/nologin -c "ConfoRL user" conforl

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=conforl:conforl /app ./

# Create directories for data, logs, and models
RUN mkdir -p /app/data /app/logs /app/models /app/config && \
    chown -R conforl:conforl /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CONFORL_LOG_LEVEL=INFO \
    CONFORL_DATA_DIR=/app/data \
    CONFORL_MODEL_DIR=/app/models \
    CONFORL_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import conforl; print('ConfoRL OK')" || exit 1

# Switch to non-root user
USER conforl

# Expose port for API (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "-m", "conforl.cli", "--help"]