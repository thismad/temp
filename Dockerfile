FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn websockets

# Copy application files
COPY server.py .
COPY index.html .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "server.py"]
