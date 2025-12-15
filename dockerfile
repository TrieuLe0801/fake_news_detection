FROM python:3.12.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for your requirements.txt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade streamlit

# Copy application files
COPY src/ ./src
COPY app.py ./app.py

# Create a non-root user and home directory
RUN groupadd -g 1005 appgroup && \
    useradd -u 1005 -g appgroup -m appuser && \
    mkdir -p .streamlit && \
    chown -R appuser:appgroup /app /home/appuser

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8001

# Set Streamlit config directory (avoid writing to /home if needed)
ENV STREAMLIT_HOME=/app/.streamlit