# tool_services/analyze_service.py
import os
import warnings
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from typing import List # Add this line
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from dotenv import load_dotenv
import json
import traceback

# --- Use OpenAI library ---
from openai import AsyncOpenAI

# Suppress specific HuggingFace tokenizers warning if necessary
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Explicit .env Loading ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dotenv_path = os.path.join(project_root, '.env')
    print(f"DEBUG (analyze_service): Attempting to load .env from: {dotenv_path}")
    if not os.path.exists(dotenv_path):
        print(f"ERROR (analyze_service): .env file check failed.")
        raise FileNotFoundError(f".env file not found at expected location: {dotenv_path}")
    else:
        print(f"DEBUG (analyze_service): Found .env file. Loading variables...")
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print("DEBUG (analyze_service): Finished loading .env file.")
except Exception as e:
    print(f"CRITICAL ERROR (analyze_service) during .env loading: {e}")
    raise RuntimeError(f"Failed during .env processing: {e}") from e
# --- End .env Loading ---

# --- Get API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INTERNAL_API_KEY = os.getenv("MCP_API_KEY")

# --- Validate Keys ---
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found (after explicit load). Check .env file.")
if not INTERNAL_API_KEY:
     raise ValueError("MCP_API_KEY (for internal auth) not found (after explicit load). Check .env file.")

# --- Initialize OpenAI Client ---
# Ensure the key is passed explicitly for clarity
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
# Choose OpenAI model - gpt-4o-mini is fast and capable, gpt-4o is best quality
OPENAI_MODEL = "gpt-4o-mini"
print(f"DEBUG (analyze_service): OpenAI client initialized for model {OPENAI_MODEL}")

# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)
async def verify_api_key(auth: str = Security(api_key_header)):
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing Internal API Key")
    return True

# --- FastAPI App ---
app = FastAPI(title="Analyze Tool Service (OpenAI)")

# --- Pydantic Models ---
# Input from Orchestrator
class AnalyzeInput(BaseModel):
    text: str = Field(..., min_length=50)
    source_url: HttpUrl | str

# Output Structure expected back from this service
class Entity(BaseModel):
    name: str
    type: str # e.g., PERSON, ORG, LOC

class AnalyzeOutput(BaseModel):
    summary: str
    entities: list[Entity] = Field(default_factory=list)
    status: str = "success"
    error_message: str | None = None

# --- Pydantic Model for OpenAI Function Calling ---
# Define the structure OpenAI should return via tool call
class AnalysisResult(BaseModel):
    """Structure for returning analysis results."""
    summary: str = Field(..., description="A concise summary (2-4 sentences) of the input text.")
    entities: List[Entity] = Field(default_factory=list, description="List of key named entities (people, organizations, locations) found in the text.")

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalyzeOutput, dependencies=[Depends(verify_api_key)])
async def run_analysis(input_data: AnalyzeInput):
    """API endpoint to analyze text using OpenAI API with Tool Calling."""
    print(f"DEBUG: Received request to analyze content from {input_data.source_url}")
    error_output = AnalyzeOutput(summary="Error during analysis", entities=[], status="error")

    # Define the tool for OpenAI based on the AnalysisResult Pydantic model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_analysis",
                "description": "Records the summary and extracted entities from the text.",
                "parameters": AnalysisResult.model_json_schema() # Use Pydantic schema
            }
        }
    ]

    # Construct messages for OpenAI
    messages = [
        {"role": "system", "content": "You are an expert text analyzer. Analyze the provided text, generate a concise summary, extract key named entities (PERSON, ORG, LOC), and call the 'record_analysis' tool with the results."},
        {"role": "user", "content": f"Source: {input_data.source_url}\n\nText to analyze:\n```\n{input_data.text}\n```"}
    ]

    try:
        print(f"DEBUG: Calling OpenAI API ({OPENAI_MODEL})...")
        completion = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "record_analysis"}}, # Force calling our tool
            temperature=0.2,
            max_tokens=1500, # Adjust if needed
            timeout=90.0
        )

        print("DEBUG: OpenAI API call returned.")
        response_message = completion.choices[0].message

        # Check if the model made the tool call as requested
        if response_message.tool_calls and response_message.tool_calls[0].function.name == "record_analysis":
            tool_call = response_message.tool_calls[0]
            print(f"DEBUG: OpenAI responded with tool call: {tool_call.function.name}")
            try:
                # Parse the arguments string (which should be JSON)
                arguments = json.loads(tool_call.function.arguments)
                print(f"DEBUG: Parsed tool arguments: {arguments}")

                # Validate the arguments using the Pydantic model
                analysis_data = AnalysisResult.model_validate(arguments)

                # Return the successful analysis result
                return AnalyzeOutput(
                    summary=analysis_data.summary,
                    entities=analysis_data.entities,
                    status="success"
                )
            except (json.JSONDecodeError, ValidationError) as e:
                msg = f"Failed to parse/validate arguments from OpenAI tool call: {e}. Arguments string: {tool_call.function.arguments}"
                print(f"ERROR: {msg}")
                error_output.error_message = msg
                return error_output
            except Exception as e_inner:
                msg = f"Error processing validated OpenAI tool call arguments: {e_inner}"
                print(f"ERROR: {msg}")
                traceback.print_exc()
                error_output.error_message = msg
                return error_output
        else:
            # Model didn't make the expected tool call
            msg = "OpenAI did not make the expected 'record_analysis' tool call."
            print(f"ERROR: {msg} - Response content: {response_message.content}")
            error_output.error_message = msg
            return error_output

    except Exception as e:
        # Catch potential API errors (rate limits, connection issues, etc.)
        # Note: openai library raises specific exceptions like openai.RateLimitError etc.
        msg = f"Error calling OpenAI API: {e}"
        print(f"ERROR: {msg}")
        traceback.print_exc()
        error_output.error_message = msg
        return error_output

# --- Health Check ---
@app.get("/health")
async def health():
    # Basic check: Can we initialize the client? (Already done above)
    # Optionally add a quick test call like listing models if needed
    return {"status": "ok"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("MCP_API_KEY"):
         print("ERROR: API Keys not loaded. Cannot run directly without .env being found.")
    else:
        uvicorn.run("analyze_service:app", host="127.0.0.1", port=8003, reload=True)