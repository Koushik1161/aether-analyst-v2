# tool_services/scrape_service.py
import os
import asyncio
import traceback # Add this line
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError

# --- Explicit .env Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True)
# --- End .env Loading ---

# Now read the variables
INTERNAL_API_KEY = os.getenv("MCP_API_KEY") # Internal auth key

# Continue with the checks...
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
app = FastAPI(title="Scrape Tool Service (Playwright)")

# --- Pydantic Models ---
class ScrapeInput(BaseModel):
    url: HttpUrl
    # Consider adding parameters for wait times, JS execution toggles later
    timeout: int = Field(30, ge=10, le=120, description="Timeout in seconds (10-120)") # Increased default/max for Playwright

class ScrapeOutput(BaseModel):
    url: str
    title: str | None
    content: str | None # Extracted text content
    status: str = "success"
    error_message: str | None = None

# --- API Endpoint ---
@app.post("/scrape", response_model=ScrapeOutput, dependencies=[Depends(verify_api_key)])
async def run_scrape_playwright(input_data: ScrapeInput):
    """API endpoint for webpage scraping using Playwright and BeautifulSoup."""
    url_str = str(input_data.url)
    print(f"Attempting to scrape URL with Playwright: {url_str}")
    error_output = ScrapeOutput(url=url_str, title=None, content=None, status="error")
    playwright = None # Define outside try block for finally
    browser = None

    try:
        playwright = await async_playwright().start()
        # Launch browser (headless is default for servers)
        # Consider 'firefox' or 'webkit' if chromium has issues with a site
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            # Optionally set user agent, viewport etc. for stealth later
             user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Navigate to the page
        print(f"Navigating to {url_str}...")
        response = await page.goto(url_str, timeout=input_data.timeout * 1000, wait_until='domcontentloaded') # Use DOM loaded initially

        # Check HTTP status from Playwright's response object
        if response is None: # Should not happen if goto succeeds without error, but good practice
            raise PlaywrightError(f"Playwright navigation returned None response for {url_str}")
        if not response.ok: # Checks for status codes 200-299
             msg = f"Received non-OK HTTP status {response.status} for {url_str}."
             print(f"ERROR: {msg}")
             error_output.error_message = msg
             return error_output # Return specific error if possible

        # Optional: Wait for potential dynamic content loading (adjust as needed)
        # await page.wait_for_timeout(3000) # Wait 3 seconds
        # Or wait for a specific element: await page.wait_for_selector("#main-content", timeout=10000)

        print(f"Page loaded. Extracting content...")
        html_content = await page.content()
        title = await page.title()
        print(f"Extracted title: {title}")

        # Close browser resources
        await context.close()
        await browser.close()
        await playwright.stop()
        browser = None # Ensure cleanup doesn't try again
        playwright = None

        # Parse the HTML obtained by Playwright using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove clutter tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript", "button", "input", "textarea"]):
            element.decompose()
        # Extract text from main content area
        main_content = soup.find("main") or soup.find("article") or soup.body
        body = None
        if main_content:
            body = main_content.get_text(separator='\n', strip=True)
        print(f"Content extracted. Length: {len(body) if body else 0}")

        return ScrapeOutput(url=url_str, title=title if title else None, content=body if body else None, status="success")

    except PlaywrightTimeoutError as e:
        msg = f"Playwright timed out loading {url_str}: {e}"
        print(f"ERROR: {msg}")
        error_output.error_message = msg
        return error_output
    except PlaywrightError as e:
        # Catch Playwright-specific errors (navigation, context creation etc.)
        msg = f"Playwright error scraping {url_str}: {e}"
        print(f"ERROR: {msg}")
        error_output.error_message = msg
        return error_output
    except Exception as e:
        # Catch other unexpected errors
        msg = f"Unexpected error during Playwright scrape of {url_str}: {e}"
        print(f"ERROR: {msg}")
        traceback.print_exc() # Log full traceback for unexpected errors
        error_output.error_message = msg
        return error_output
    finally:
        # Ensure browser resources are closed even if errors occur
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()


# --- Health Check ---
@app.get("/health")
async def health():
    # Basic health check, could add a quick Playwright launch test if needed
    return {"status": "ok"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    if not os.getenv("MCP_API_KEY"):
         print("ERROR: MCP_API_KEY not loaded. Cannot run directly without .env being found.")
    else:
        # Run with string path for consistency if needed by some runners
        uvicorn.run("scrape_service:app", host="127.0.0.1", port=8002, reload=True)