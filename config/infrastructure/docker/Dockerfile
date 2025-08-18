FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz | tar xz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet vader_lexicon

# Copy application code
COPY backend /app/backend
COPY models /app/models

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Default command
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]