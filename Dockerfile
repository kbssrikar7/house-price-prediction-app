# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Install dependencies
RUN uv sync

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports figures

# Default command
CMD ["uv", "run", "python", "src/train_supervised.py"]
