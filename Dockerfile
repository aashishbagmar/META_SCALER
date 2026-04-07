FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY env.py .
COPY models.py .
COPY grader.py .
COPY inference.py .
COPY openenv.yaml .

# Set environment variables with defaults
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=qwen2.5-72b-instruct

# Run inference script
CMD ["python", "inference.py"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]