# test_embedding.py
from sentence_transformers import SentenceTransformer
import time

model_name = 'all-MiniLM-L6-v2'
print(f"Attempting to download/load embedding model: {model_name}...")
start_time = time.time()
try:
    # This will download the model to the cache if not present
    model = SentenceTransformer(model_name)
    end_time = time.time()
    print(f"Model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds!")
except Exception as e:
    end_time = time.time()
    print(f"Failed to load model after {end_time - start_time:.2f} seconds: {e}")