FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn websockets orjson

# Copy application files
COPY server.py .
COPY index.html .

# Expose port
EXPOSE 8000

# Production: no bots by default
ENV NUM_BOTS=0

# Run the application
CMD ["python", "server.py"]
