# Custom Airflow Docker image for Investment Analysis Application
# Based on official Apache Airflow image with additional dependencies

FROM apache/airflow:2.7.3-python3.11

# Switch to root to install system packages
USER root

# Install system dependencies required for financial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libyaml-dev \
    libhdf5-dev \
    pkg-config \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libatlas-base-dev \
    wget \
    curl \
    git \
    nano \
    htop \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (Technical Analysis Library) from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set environment variables for Python packages
ENV PYTHONPATH=/opt/airflow:/opt/airflow/backend:/opt/airflow/models
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__PLUGINS_FOLDER=/opt/airflow/plugins

# Switch back to airflow user
USER airflow

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies for investment analysis
COPY requirements-airflow.txt /opt/airflow/requirements-airflow.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements-airflow.txt

# Install additional Python packages that might be needed
RUN pip install --no-cache-dir \
    pendulum==2.1.2 \
    apache-airflow-providers-slack==8.3.0 \
    apache-airflow-providers-amazon==8.8.0 \
    apache-airflow-providers-google==10.10.1 \
    great-expectations==0.18.1 \
    dbt-core==1.6.7 \
    dbt-postgres==1.6.7 \
    mlflow==2.8.1

# Create necessary directories
RUN mkdir -p /opt/airflow/dags \
             /opt/airflow/logs \
             /opt/airflow/plugins \
             /opt/airflow/config \
             /opt/airflow/backend \
             /opt/airflow/models

# Copy custom plugins if any exist
# COPY --chown=airflow:airflow plugins/ /opt/airflow/plugins/

# Copy configuration files
COPY --chown=airflow:airflow data_pipelines/airflow/config/ /opt/airflow/config/

# Set proper permissions
USER root
RUN chown -R airflow:airflow /opt/airflow
USER airflow

# Create a startup script for initialization
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to wait for database\n\
wait_for_db() {\n\
    echo "Waiting for database..."\n\
    while ! nc -z ${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN%/*} 5432; do\n\
        sleep 1\n\
    done\n\
    echo "Database is ready!"\n\
}\n\
\n\
# Function to initialize database if needed\n\
init_db_if_needed() {\n\
    if [[ "${AIRFLOW_ROLE:-}" == "webserver" || "${AIRFLOW_ROLE:-}" == "scheduler" ]]; then\n\
        echo "Checking if database needs initialization..."\n\
        if ! airflow db check 2>/dev/null; then\n\
            echo "Initializing Airflow database..."\n\
            airflow db init\n\
            echo "Database initialized!"\n\
        else\n\
            echo "Database already initialized."\n\
            echo "Running database upgrade..."\n\
            airflow db upgrade\n\
        fi\n\
    fi\n\
}\n\
\n\
# Function to create admin user\n\
create_admin_user() {\n\
    if [[ "${AIRFLOW_ROLE:-}" == "webserver" ]]; then\n\
        echo "Creating admin user if not exists..."\n\
        airflow users create \\\n\
            --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \\\n\
            --firstname "Investment" \\\n\
            --lastname "Admin" \\\n\
            --role "Admin" \\\n\
            --email "${AIRFLOW_ADMIN_EMAIL:-admin@investment-analysis.com}" \\\n\
            --password "${AIRFLOW_ADMIN_PASSWORD:-admin123}" \\\n\
            2>/dev/null || echo "Admin user already exists or creation failed."\n\
    fi\n\
}\n\
\n\
# Main execution\n\
if [[ "${1:-}" == "webserver" || "${1:-}" == "scheduler" || "${1:-}" == "worker" ]]; then\n\
    export AIRFLOW_ROLE="${1}"\n\
    wait_for_db\n\
    init_db_if_needed\n\
    create_admin_user\n\
fi\n\
\n\
# Execute the original command\n\
exec "$@"\n' > /opt/airflow/entrypoint-custom.sh

# Make the script executable
USER root
RUN chmod +x /opt/airflow/entrypoint-custom.sh
USER airflow

# Health check script
RUN echo '#!/bin/bash\n\
case "${AIRFLOW_ROLE:-}" in\n\
    "webserver")\n\
        curl -f http://localhost:8080/health || exit 1\n\
        ;;\n\
    "scheduler")\n\
        airflow jobs check --job-type SchedulerJob --hostname "${HOSTNAME}" || exit 1\n\
        ;;\n\
    "worker")\n\
        celery --app airflow.executors.celery_executor.app inspect ping -d "celery@${HOSTNAME}" || exit 1\n\
        ;;\n\
    *)\n\
        echo "Unknown role: ${AIRFLOW_ROLE:-}"\n\
        exit 1\n\
        ;;\n\
esac\n' > /opt/airflow/healthcheck.sh

USER root
RUN chmod +x /opt/airflow/healthcheck.sh
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Labels for the image
LABEL maintainer="Investment Analysis Team"
LABEL version="1.0.0"
LABEL description="Custom Airflow image for investment analysis with financial libraries"
LABEL airflow.version="2.7.3"
LABEL python.version="3.11"

# Expose port
EXPOSE 8080

# Use custom entrypoint
ENTRYPOINT ["/opt/airflow/entrypoint-custom.sh"]
CMD ["airflow", "webserver"]