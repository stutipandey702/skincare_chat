# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy application files
COPY . /app

# Create and set up Hugging Face cache directory
RUN mkdir -p /tmp/hf_cache_lora
ENV HF_HOME=/tmp/hf_cache_lora \
    TRANSFORMERS_CACHE=/tmp/hf_cache_lora \
    HF_DATASETS_CACHE=/tmp/hf_cache_lora \
    HF_METRICS_CACHE=/tmp/hf_cache_lora

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Run the Flask app
CMD ["flask", "run"]
