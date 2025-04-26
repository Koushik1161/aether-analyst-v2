# Dockerfile (Root - for Streamlit UI + Orchestrator)
FROM python:3.11-slim

WORKDIR /app

# Install git (needed by some langchain/sentence-transformer dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- ADD THIS LINE ---
# Explicitly install streamlit to be sure
RUN pip install --no-cache-dir streamlit==1.44.1
# --- END ADDED LINE ---

# Copy orchestrator, UI, and utils code
# Ensure destination directories end with '/' when copying directories
COPY ./orchestrator /app/orchestrator/
COPY ./main.py /app/main.py
# COPY ./utils /app/utils/ # Uncomment if you have utils
# COPY ./utils/__init__.py /app/utils/__init__.py # If needed

# Copy the .env file (For demo purposes; use env_file in compose for better practice)
COPY ./.env /app/.env

# Expose the Streamlit port
EXPOSE 8501

# Set HF cache directory for embedding model loaded by orchestrator
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Command to run Streamlit
# Use --server.address=0.0.0.0 to make it accessible outside the container
CMD ["python", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]