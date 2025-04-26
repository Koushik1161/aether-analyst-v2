# tool_services/processing_service.py
import os
import warnings
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# Suppress specific HuggingFace tokenizers warning if necessary
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Explicit .env Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
# --- End .env Loading ---

# Now read the variables
INTERNAL_API_KEY = os.getenv("MCP_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Uncomment if using Qdrant Cloud

# Continue with the checks...
if not INTERNAL_API_KEY:
     raise ValueError("MCP_API_KEY (for internal auth) not found (after explicit load).")

# --- Config ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_COLLECTION_NAME = "aether_analyst_docs"

# --- Initialize Clients and Models (Load only once on startup) ---
# Wrap in a function to handle potential errors during startup
def initialize_components():
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
        EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {EMBEDDING_DIM}")
    except Exception as e:
        print(f"FATAL: Failed to load embedding model '{EMBEDDING_MODEL_NAME}'. Error: {e}")
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    try:
        print(f"Connecting to Qdrant at {QDRANT_URL}...")
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60)
        print("Qdrant client created.")
        # Ensure collection exists
        try:
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")
        except Exception:
             print(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
             qdrant_client.recreate_collection(
                 collection_name=QDRANT_COLLECTION_NAME,
                 vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
             )
             print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
        print("Qdrant setup complete.")
    except Exception as e:
         print(f"FATAL: Failed to connect to or setup Qdrant. Error: {e}")
         raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e

    return embedding_model, qdrant_client, EMBEDDING_DIM

embedding_model, qdrant_client, EMBEDDING_DIM = initialize_components()

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, length_function=len,
    is_separator_regex=False, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)
async def verify_api_key(auth: str = Security(api_key_header)):
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return True

# --- FastAPI App ---
app = FastAPI(title="Processing Service (Chunking, Embedding, Storing)")

# --- Pydantic Models ---
class ProcessInput(BaseModel):
    content: str = Field(..., min_length=10)
    source_url: str

class ProcessOutput(BaseModel):
    status: str = "success"
    message: str
    chunks_processed: int
    error_message: str | None = None

# --- API Endpoint ---
@app.post("/process", response_model=ProcessOutput, dependencies=[Depends(verify_api_key)])
async def process_and_store(input_data: ProcessInput):
    """Chunks text, generates embeddings, and stores in Qdrant."""
    print(f"Processing content from: {input_data.source_url}")
    error_output = ProcessOutput(status="error", message="Processing failed", chunks_processed=0)
    try:
        chunks = text_splitter.split_text(input_data.content)
        if not chunks:
            return ProcessOutput(status="success", message="No content chunks to process.", chunks_processed=0)
        print(f"Generated {len(chunks)} chunks.")

        # Generate Embeddings
        # Note: encode can be CPU intensive; run in executor if blocking async loop
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
        print(f"Generated {len(embeddings)} embeddings.")
        if len(embeddings) != len(chunks):
            raise ValueError("Mismatch between chunks and embeddings count.")

        # Prepare points
        points_to_upsert = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={"text": chunk, "source": input_data.source_url, "chunk_index": i}
            ) for i, chunk in enumerate(chunks)
        ]

        # Upsert into Qdrant
        if points_to_upsert:
            print(f"Upserting {len(points_to_upsert)} points to Qdrant...")
            response = qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points_to_upsert,
                wait=True
            )
            print(f"Qdrant upsert response: {response.status}")
            if response.status != models.UpdateStatus.COMPLETED:
                 raise RuntimeError(f"Qdrant upsert failed: {response.status}")

        return ProcessOutput(
            status="success",
            message=f"Successfully processed and stored content from {input_data.source_url}",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        msg = f"Error during processing/storing for {input_data.source_url}: {e}"
        print(f"ERROR: {msg}")
        import traceback
        traceback.print_exc() # Log full traceback for unexpected errors
        error_output.error_message = str(e) # Provide error message in response
        return error_output

# --- Health Check ---
@app.get("/health")
async def health():
    # Basic check: Can we reach Qdrant?
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "error_connecting"
    return {"status": "ok", "qdrant_status": qdrant_status}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    if not os.getenv("MCP_API_KEY"):
         print("ERROR: MCP_API_KEY not loaded. Cannot run directly without .env being found.")
    else:
        # Consider adding host='0.0.0.0' if running in Docker later
        uvicorn.run(app, host="127.0.0.1", port=8005) # Match port in .env