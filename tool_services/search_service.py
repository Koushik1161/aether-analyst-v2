# tool_services/search_service.py
import os
import httpx
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# --- Explicit .env Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
# --- End .env Loading ---

# Now read the variables
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
INTERNAL_API_KEY = os.getenv("MCP_API_KEY") # Use the key meant for internal auth

# Continue with the checks...
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not found in environment variables (after explicit load).")
if not INTERNAL_API_KEY:
     raise ValueError("MCP_API_KEY (for internal auth) not found (after explicit load).")

# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)
async def verify_api_key(auth: str = Security(api_key_header)):
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return True

# --- FastAPI App ---
app = FastAPI(title="Search Tool Service")

# --- Pydantic Models ---
class SearchInput(BaseModel):
    query: str = Field(..., description="The search query string.")
    num_results: int = Field(5, ge=1, le=20, description="Number of results (1-20).")

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str | None

class SearchOutput(BaseModel):
    results: list[SearchResult]
    status: str = "success"
    error_message: str | None = None

# --- API Endpoint ---
@app.post("/search", response_model=SearchOutput, dependencies=[Depends(verify_api_key)])
async def run_search(input_data: SearchInput):
    """API endpoint to perform a web search using SerpAPI."""
    params = {
        "q": input_data.query,
        "num": input_data.num_results,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"
    }
    error_result = SearchOutput(results=[], status="error")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://serpapi.com/search", params=params, timeout=15.0)
            response.raise_for_status() # Raise HTTP errors
            data = response.json()
            items = data.get("organic_results", [])
            results = []
            for item in items[:input_data.num_results]:
                try:
                    results.append(SearchResult(
                        title=item.get("title", "N/A"),
                        url=item.get("link", ""),
                        snippet=item.get("snippet")
                    ))
                except (ValidationError, KeyError) as e:
                    print(f"Warning: Skipping search result due to error: {e}")
            return SearchOutput(results=results, status="success")
    except httpx.RequestError as e:
        msg = f"Error connecting to SerpAPI: {e}"
        print(f"ERROR: {msg}")
        error_result.error_message = msg
        return error_result
    except httpx.HTTPStatusError as e:
        msg = f"SerpAPI request failed: {e.response.status_code}"
        print(f"ERROR: {msg} - {e.response.text}")
        error_result.error_message = msg
        return error_result
    except Exception as e:
        msg = f"Unexpected error in search service: {e}"
        print(f"ERROR: {msg}")
        error_result.error_message = msg
        return error_result

# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Uvicorn Runner (for direct testing if needed) ---
if __name__ == "__main__":
    import uvicorn
    # Note: Ensure .env is loaded correctly even when run directly
    if not os.getenv("SERPAPI_API_KEY"):
         print("ERROR: SERPAPI_API_KEY not loaded. Cannot run directly without .env being found.")
    else:
         uvicorn.run(app, host="0.0.0.0", port=8001) # Match port in .env