# Explicitly specify AMD64 platform
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Set non-sensitive environment variables
ENV PYTHONUNBUFFERED=1
ENV S3_BUCKET_NAME=semantic-s3
ENV ATHENA_OUTPUT_BUCKET=s3://semantic-s3/query-results/

# Increase pip timeout and retries
RUN pip config set global.timeout 1000 && \
    pip config set global.retries 10

# Add more reliable PyPI mirrors
RUN pip config set global.index-url https://pypi.org/simple/ && \
    pip config set global.extra-index-url https://pypi.python.org/simple/

# Install system dependencies for pyarrow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wave_processor.py .
COPY parquet_creator.py .

# Command to run the script
ENTRYPOINT ["python", "wave_processor.py"]