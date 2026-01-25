# Production-Ready Backend Dockerfile
# Multi-stage build for optimal size and security

# Build args
ARG PYTHON_VERSION=3.12
ARG BUILD_ENV=production

# Stage 1: Builder
FROM python:${PYTHON_VERSION}-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (required for technical analysis)
# Detect architecture and configure appropriately
RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz | tar xz && \
    cd ta-lib && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        ./configure --prefix=/usr/local --build=aarch64-unknown-linux-gnu; \
    else \
        ./configure --prefix=/usr/local; \
    fi && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords wordnet vader_lexicon

# Stage 2: Development (for hot-reloading during development)
FROM python:${PYTHON_VERSION}-slim AS development

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy TA-Lib libraries from builder
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib /usr/local/include/ta-lib

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy NLTK data
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Update library cache
RUN ldconfig

# Set environment variables for development
ENV PYTHONPATH=/app:/root/.local/lib/python3.12/site-packages \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    NLTK_DATA=/usr/local/share/nltk_data \
    DEBUG=true

# Expose port
EXPOSE 8000

# Development command (overridden by docker-compose)
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Runtime (Production)
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy TA-Lib libraries from builder
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib /usr/local/include/ta-lib

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy NLTK data
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Copy application code
COPY --chown=appuser:appuser backend /app/backend
COPY --chown=appuser:appuser models /app/models
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser alembic.ini /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/ml_models /app/cache && \
    chown -R appuser:appuser /app

# Update library cache
RUN ldconfig

# Set environment variables
ENV PYTHONPATH=/app:/home/appuser/.local/lib/python3.12/site-packages \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    NLTK_DATA=/usr/local/share/nltk_data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Default command - can be overridden
CMD ["gunicorn", "backend.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50"]