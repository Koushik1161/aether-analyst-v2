# tool_services/news_service.py
import os
import httpx
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from dotenv import load_dotenv
from datetime import datetime

# --- Explicit .env Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
# --- End .env Loading ---

# Now read the variables
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
INTERNAL_API_KEY = os.getenv("MCP_API_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Continue with the checks...
if not NEWSAPI_API_KEY:
    raise ValueError("NEWSAPI_API_KEY not found (after explicit load).")
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
app = FastAPI(title="News Tool Service")

# --- Pydantic Models ---
class NewsInput(BaseModel):
    topic: str = Field(...)
    max_articles: int = Field(5, ge=1, le=20)

class Article(BaseModel):
    title: str
    url: HttpUrl
    # Use string initially, validation/parsing happens during model creation
    published_at: str | datetime
    source: str
    snippet: str | None

class NewsOutput(BaseModel):
    articles: list[Article]
    status: str = "success"
    error_message: str | None = None

# --- API Endpoint ---
@app.post("/news", response_model=NewsOutput, dependencies=[Depends(verify_api_key)])
async def run_news(input_data: NewsInput):
    """API endpoint to fetch recent news articles using NewsAPI."""
    params = {"q": input_data.topic, "pageSize": input_data.max_articles, "apiKey": NEWSAPI_API_KEY, "sortBy": "publishedAt"}
    headers = {"User-Agent": "AetherAnalyst/1.0"}
    error_output = NewsOutput(articles=[], status="error")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(NEWSAPI_URL, params=params, headers=headers, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "ok":
                msg = f"NewsAPI error: {data.get('message', 'Unknown')}"
                print(f"ERROR: {msg}")
                error_output.error_message = msg
                return error_output

            raw_articles = data.get("articles", [])
            articles = []
            for item in raw_articles:
                try:
                    # Attempt to parse datetime, fallback to string if invalid
                    published_dt = None
                    try:
                         published_dt = datetime.fromisoformat(item.get("publishedAt").replace('Z', '+00:00')) # Handle Z timezone
                    except (ValueError, TypeError, AttributeError):
                         published_dt = item.get("publishedAt", "N/A") # Keep as string if parsing fails

                    articles.append(Article(
                        title=item.get("title", "N/A"),
                        url=item.get("url"), # Pydantic validates HttpUrl
                        published_at=published_dt,
                        source=item.get("source", {}).get("name", "N/A"),
                        snippet=item.get("description")
                    ))
                except ValidationError as e:
                    print(f"Warning: Skipping news article due to validation error: {e}")
                except KeyError as e:
                     print(f"Warning: Skipping news article due to missing key: {e}")
            return NewsOutput(articles=articles, status="success")

    except httpx.RequestError as e:
        msg = f"Error contacting NewsAPI: {e}"
        print(f"ERROR: {msg}")
        error_output.error_message = msg
        return error_output
    except httpx.HTTPStatusError as e:
        msg = f"NewsAPI request failed: {e.response.status_code}"
        print(f"ERROR: {msg} - {e.response.text}")
        error_output.error_message = msg
        return error_output
    except Exception as e:
        msg = f"Unexpected error in news service: {e}"
        print(f"ERROR: {msg}")
        error_output.error_message = msg
        return error_output

# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    if not os.getenv("NEWSAPI_API_KEY") or not os.getenv("MCP_API_KEY"):
         print("ERROR: API Keys not loaded. Cannot run directly without .env being found.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8004) # Match port in .env