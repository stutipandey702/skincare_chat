FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create cache directories with proper permissions
RUN mkdir -p /app/hf_cache && \
    chmod -R 777 /app/hf_cache && \
    mkdir -p /app/models && \
    chmod -R 777 /app/models

# Set all HF environment variables to use /app
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/app/hf_cache

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Make sure all directories are writable
RUN chmod -R 777 /app

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]