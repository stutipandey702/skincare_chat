# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy your application code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed by your app
RUN python -m nltk.downloader punkt

# Expose the port your Flask app runs on
EXPOSE 8080

# Set environment variables (optional: can be overridden at runtime)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Run the Flask app
CMD ["flask", "run"]
