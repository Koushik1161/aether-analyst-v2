# orchestrator/agent_graph.py
import os
import httpx
import json
# Ensure all necessary types are imported
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Set
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
# Using Pydantic v1 namespace for compatibility as suggested by warning,
# but consider migrating fully to v2 later if needed.
from langchain_core.pydantic_v1 import BaseModel, Field

# Tool definition and execution helpers
from langchain_core.tools import tool
# USE the standard ToolNode
from langgraph.prebuilt import ToolNode

# Graph building
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# RAG components
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import uuid
import traceback
import warnings
import asyncio

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pydantic v1.*")

# --- Explicit .env Loading with Absolute Path & Debugging ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dotenv_path = os.path.join(project_root, '.env')
    print(f"DEBUG (agent_graph): Attempting to load .env from: {dotenv_path}")
    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found at expected location: {dotenv_path}")
    else:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print("DEBUG (agent_graph): Finished loading .env file.")
except Exception as e:
    raise RuntimeError(f"Failed during .env processing: {e}") from e
# --- End .env Loading ---

# --- Get API Keys and Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INTERNAL_API_KEY = os.getenv("MCP_API_KEY") # Key for calling tool services

TOOL_SEARCH_URL = os.getenv("TOOL_SEARCH_URL", "http://localhost:8001")
TOOL_SCRAPE_URL = os.getenv("TOOL_SCRAPE_URL", "http://localhost:8002")
TOOL_ANALYZE_URL = os.getenv("TOOL_ANALYZE_URL", "http://localhost:8003")
TOOL_NEWS_URL = os.getenv("TOOL_NEWS_URL", "http://localhost:8004")
TOOL_PROCESS_URL = os.getenv("TOOL_PROCESS_URL", "http://localhost:8005")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
QDRANT_COLLECTION_NAME = "aether_analyst_docs"

# --- Validate Keys ---
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not found.")
if not INTERNAL_API_KEY: raise ValueError("MCP_API_KEY (for internal auth) not found.")

# --- Initialize RAG Components ---
try:
    print(f"Orchestrator: Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Orchestrator: Connecting to Qdrant at {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60)
    print("Embedding model and Qdrant client loaded for orchestrator.")
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Orchestrator: Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception as e:
        print(f"Warning: Qdrant collection '{QDRANT_COLLECTION_NAME}' not found by orchestrator during startup check. Error: {e}")
except Exception as e:
     raise RuntimeError(f"Failed RAG component init in orchestrator: {e}") from e

# --- Define LangGraph State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    scraped_data: Dict[str, Dict[str, Any]] # url -> {"content": str, "title": str}
    processed_urls: Set[str] # Stores URLs sent for processing
    analysis_results: Dict[str, Dict[str, Any]] # url -> {"summary": str, "entities": list}
    news_results: Optional[List[Dict[str, Any]]] = None # Store news results if fetched
    retrieved_context: Optional[str] = None
    final_report: Optional[str] = None # Store the final generated report
    error_log: List[str] = []

# --- Define Tool Input Schemas ---
class SearchToolInput(BaseModel): query: str = Field(...); num_results: int = Field(5)
class ScrapeToolInput(BaseModel): url: str = Field(...)
class ProcessToolInput(BaseModel): content: str = Field(...); source_url: str = Field(...)
class AnalyzeToolInput(BaseModel): text: str = Field(...); source_url: str = Field(...)
class NewsToolInput(BaseModel): topic: str = Field(...); max_articles: int = Field(5)

# --- Tool Calling Function ---
async def call_service_api(service_url: str, endpoint: str, payload: dict) -> dict:
    """Helper to call a tool microservice."""
    headers = {"Authorization": f"Bearer {INTERNAL_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    url = f"{service_url}{endpoint}"
    print(f"Calling Tool Endpoint: POST {url} with payload keys: {list(payload.keys())}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e: error_msg = f"Network error calling {endpoint}: {e}"; print(f"ERROR: {error_msg}"); return {"status": "error", "error_message": error_msg}
        except httpx.HTTPStatusError as e: error_msg = f"Tool service error for {endpoint} ({e.response.status_code}): {e.response.text[:500]}"; print(f"ERROR: {error_msg}"); return {"status": "error", "error_message": error_msg}
        except Exception as e: error_msg = f"Unexpected error calling {endpoint}: {e}"; print(f"ERROR: {error_msg}"); return {"status": "error", "error_message": error_msg}

# --- Tool Definitions (@tool decorator) ---
@tool(args_schema=SearchToolInput)
async def search_web(query: str, num_results: int = 5) -> str:
    """Searches the web for relevant documents based on the query. Use this first to find relevant URLs for research. Returns JSON results including 'status'."""
    result = await call_service_api(TOOL_SEARCH_URL, "/search", {"query": query, "num_results": num_results}); return json.dumps(result)

@tool(args_schema=ScrapeToolInput)
async def scrape_webpage(url: str) -> str:
    """Scrapes the text content from a given webpage URL using Playwright. MUST be called for relevant URLs found by search_web. Returns JSON including 'status', 'content', 'error_message'."""
    result = await call_service_api(TOOL_SCRAPE_URL, "/scrape", {"url": url}); return json.dumps(result)

@tool(args_schema=ProcessToolInput)
async def process_and_store_content(content: str, source_url: str) -> str:
    """Processes successfully scraped text content: chunks, embeds, and stores it in the Qdrant vector database for later retrieval. MUST be called AFTER successfully scraping content with scrape_webpage. Returns JSON including 'status'."""
    if not content or len(content) < 50: print("Skipping process_and_store_content: Content too short."); return json.dumps({"status": "skipped", "message": "Content too short or empty.", "chunks_processed": 0})
    result = await call_service_api(TOOL_PROCESS_URL, "/process", {"content": content, "source_url": source_url}); return json.dumps(result)

@tool(args_schema=AnalyzeToolInput)
async def analyze_content(text: str, source_url: str) -> str:
    """Summarizes text and extracts named entities (people, places, orgs) using the OpenAI API. Use this AFTER processing content, ideally on specific relevant chunks found via retrieval, or on full texts if needed early."""
    if not text or len(text) < 50: print("Skipping analyze_content: Text too short."); return json.dumps({"status": "skipped", "message": "Analyze content too short or empty."})
    result = await call_service_api(TOOL_ANALYZE_URL, "/analyze", {"text": text, "source_url": source_url}); return json.dumps(result)

@tool(args_schema=NewsToolInput)
async def aggregate_news(topic: str, max_articles: int = 5) -> str:
    """Fetches recent news articles on a specific topic using NewsAPI. Use ONLY if the query specifically asks for recent news. Returns JSON including 'status', 'articles'."""
    result = await call_service_api(TOOL_NEWS_URL, "/news", {"topic": topic, "max_articles": max_articles}); return json.dumps(result)

# --- List of Tools ---
tools = [search_web, scrape_webpage, process_and_store_content, analyze_content, aggregate_news]

# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)
synthesis_llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Separate instance for synthesis

# --- Graph Nodes ---

def agent_node(state: AgentState) -> Dict[str, Any]:
    """Processes tool results, checks for processing needs, calls LLM for next action."""
    print("--- Agent Node: Entered ---")
    messages = state["messages"]
    processed_urls = state.get("processed_urls", set()).copy()
    scraped_data_map = state.get("scraped_data", {}).copy()
    analysis_results_map = state.get("analysis_results", {}).copy()
    news_results_list = state.get("news_results")
    if news_results_list is not None: news_results_list = news_results_list[:] # Shallow copy list
    error_log_list = state.get("error_log", [])[:]

    next_step_tool_calls = []

    # --- Process Last Tool Message & Update State ---
    if messages and isinstance(messages[-1], ToolMessage):
        last_tool_message = messages[-1]
        print(f"--- Agent Node: Processing result from Tool '{last_tool_message.name}' ---")
        tool_name = last_tool_message.name
        tool_content_str = last_tool_message.content
        try:
            tool_result = json.loads(tool_content_str)
            tool_status = tool_result.get("status")
            tool_error = tool_result.get("error_message")

            if tool_status == "error":
                 print(f"ERROR reported by tool '{tool_name}': {tool_error}")
                 error_log_list.append(f"Tool '{tool_name}' failed: {tool_error}")
            # --- Update State Based on Tool ---
            elif tool_name == "scrape_webpage":
                url = tool_result.get("url")
                content = tool_result.get("content")
                if url and content: # Successful scrape with content
                    if url not in processed_urls: # Check if already processed
                         print(f"DEBUG (Agent Node): Scrape success for {url}. Preparing to process.")
                         scraped_data_map[url] = {"content": content, "title": tool_result.get("title")}
                         # We will generate the process call below
                    else:
                        print(f"DEBUG (Agent Node): Scrape success for {url}, but already processed.")
                elif url: # Scrape might succeed but return no content
                    print(f"DEBUG (Agent Node): Scrape for {url} returned status '{tool_status}' but no content.")
                    error_log_list.append(f"Scrape for {url} returned no content.")
            elif tool_name == "process_and_store_content":
                 source_url = tool_result.get("source_url") # Assuming service returns this
                 if source_url and tool_status == "success":
                    print(f"DEBUG: Successfully processed/stored content from {source_url}")
                    processed_urls.add(source_url) # Mark as processed *after* success
                 elif source_url and tool_result.get("status") != "skipped":
                     error_log_list.append(f"Processing failed for {source_url}: {tool_result.get('message')}")
            elif tool_name == "analyze_content":
                 source_url = tool_result.get("source_url") # Analyze service needs to return this!
                 if source_url and tool_status == "success":
                      analysis_results_map[source_url] = {"summary": tool_result.get("summary"), "entities": tool_result.get("entities")}
                      print(f"DEBUG: Stored analysis result for {source_url}")
                 elif source_url:
                      error_log_list.append(f"Analysis failed for {source_url}: {tool_error}")
            elif tool_name == "aggregate_news":
                 if tool_status == "success":
                     news_results_list = tool_result.get("articles") # Overwrite previous news
                     print(f"DEBUG: Stored {len(news_results_list) if news_results_list else 0} news articles.")
                 else:
                     error_log_list.append(f"News aggregation failed: {tool_error}")

        except json.JSONDecodeError: print(f"Warning: Could not decode JSON from {tool_name} ToolMessage: {tool_content_str}"); error_log_list.append(f"Failed to decode result from {tool_name}")
        except Exception as e: print(f"Error processing ToolMessage from {tool_name}: {e}"); error_log_list.append(f"Error processing result from {tool_name}: {e}")

    # --- Check if any newly scraped data needs processing ---
    for url, data in scraped_data_map.items():
         if url not in processed_urls:
             content = data.get("content")
             if content:
                  print(f"DEBUG (Agent Node): Generating process_and_store_content call for {url}")
                  next_step_tool_calls.append(
                      # Use dict structure for tool call generation
                      {"name": "process_and_store_content", "args": {"content": content, "source_url": url}, "id": f"call_proc_{uuid.uuid4().hex[:4]}"}
                  )
                  processed_urls.add(url) # Mark as processed now we've generated the call

    # If processing calls were generated, execute them next
    if next_step_tool_calls:
        print(f"--- Agent Node: Prioritizing {len(next_step_tool_calls)} process_and_store_content call(s) ---")
        return {
            "messages": [AIMessage(content="", tool_calls=next_step_tool_calls)],
            "scraped_data": scraped_data_map, # Persist updates
            "processed_urls": processed_urls,
            "analysis_results": analysis_results_map,
            "news_results": news_results_list,
            "error_log": error_log_list
        }

    # If no processing needed, call LLM to decide next step or synthesize
    print(f"--- Agent Node: No processing needed. Calling LLM invoke... ---")
    print(f"Messages passed to LLM: {[msg.pretty_repr() for msg in messages]}")
    try:
        response = llm_with_tools.invoke(messages) # Pass full history
        print("--- Agent Node: LLM invoke returned ---")
        print(f"LLM Response content: {response.content}")
        if hasattr(response, 'tool_calls') and response.tool_calls: print(f"LLM Response tool_calls: {response.tool_calls}")
        else: print("LLM Response has no tool calls -> suggests synthesis/end.")
        # Return LLM response + updated state fields
        return {
            "messages": [response],
            "scraped_data": scraped_data_map,
            "processed_urls": processed_urls,
            "analysis_results": analysis_results_map,
            "news_results": news_results_list,
            "error_log": error_log_list
        }
    except Exception as e:
        error_msg = f"Error invoking LLM: {e}"; print(f"!!!!!!!! ERROR during LLM invoke: {error_msg} !!!!!!!!")
        traceback.print_exc(); current_errors = state.get("error_log", [])
        return {"messages": [AIMessage(content=f"Error: Could not invoke LLM. {error_msg}")], "error_log": current_errors + [error_msg]}

# Use the standard ToolNode from LangGraph prebuilt
tool_node = ToolNode(tools)

# Retrieval node
def retrieve_context_node(state: AgentState):
    """Retrieves relevant context from Qdrant based on the original query."""
    print("--- Retrieving Context Node ---")
    original_query = state.get("query")
    current_errors = state.get("error_log", [])
    retrieved_context = None
    error_log = current_errors

    if not original_query:
        print("Warning: Original query not found in state for retrieval.")
        error_log = current_errors + ["Query missing for retrieval."]
    else:
        if not state.get("processed_urls"): # Check if any URLs were processed
            print("DEBUG: No URLs processed yet, skipping retrieval.")
            retrieved_context = "No content was processed and stored in the vector database for this query."
        else:
            try:
                print(f"DEBUG: Generating embedding for query: {original_query[:100]}...")
                query_embedding = embedding_model.encode(original_query).tolist()
                print(f"DEBUG: Searching Qdrant collection '{QDRANT_COLLECTION_NAME}' (limit 5)...")
                search_result = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=query_embedding, limit=5)
                retrieved_docs = [f"Source: {hit.payload.get('source', 'Unknown')}\nContent:\n{hit.payload.get('text', '')}" for hit in search_result]
                if not retrieved_docs: context_str = "No relevant context found in the vector store for this query."; print("DEBUG: No documents found in Qdrant search.")
                else: context_str = "\n\n---\n\n".join(retrieved_docs); print(f"Retrieved {len(retrieved_docs)} chunks from Qdrant.")
                retrieved_context = context_str
            except Exception as e: error_msg = f"Error retrieving from Qdrant: {e}"; print(f"ERROR: {error_msg}"); traceback.print_exc(); retrieved_context = f"Error during retrieval: {e}"; error_log = current_errors + [error_msg]

    # Return updates to be merged into the state
    return {"retrieved_context": retrieved_context, "error_log": error_log}

# Synthesis Node
def synthesis_node(state: AgentState):
    """Generates the final report using the LLM, incorporating retrieved context."""
    print("--- Synthesis Node: Generating Final Report ---")
    query = state["query"]
    retrieved_context = state.get("retrieved_context", "No context was retrieved or retrieval failed.")
    # --- Improved Context for Synthesis ---
    # Combine Analysis results and News results into readable strings
    analysis_texts = []
    for url, result in state.get("analysis_results", {}).items():
        summary = result.get('summary', 'N/A')
        entities = ", ".join([e.get('name', 'N/A') for e in result.get('entities', [])])
        analysis_texts.append(f"Analysis of {url}:\n Summary: {summary}\n Entities: {entities}")
    analysis_summary = "\n---\n".join(analysis_texts) if analysis_texts else "No content analysis was performed."

    news_list = state.get("news_results", []) or []
    news_summary = "\n".join([f"- {a.get('title', 'N/A')} ({a.get('source', 'N/A')}) Link: {a.get('url')}" for a in news_list]) if news_list else "No news search was performed."
    # --- End Improved Context ---

    # --- Synthesis Prompt ---
    synthesis_prompt_messages = [
        HumanMessage(content=f"""You are Aether Analyst, an AI research assistant. Your task is to synthesize a comprehensive report answering the user's original query, grounded in the retrieved context from web scraping.

        **Original User Query:**
        {query}

        **Retrieved Context from Scraped Web Pages (Primary Source):**
        ```
        {retrieved_context}
        ```

        **Analysis Summaries/Entities (Optional Secondary Context):**
        ```
        {analysis_summary}
        ```

        **Recent News Articles (Optional Secondary Context):**
        ```
        {news_summary}
        ```

        **Instructions:**
        1.  Carefully review the 'Retrieved Context'. This is your primary source material.
        2.  Generate a well-structured, informative report in Markdown format that directly answers the 'Original User Query'.
        3.  Base your answer *primarily* on the information found in the 'Retrieved Context'.
        4.  You MAY use the 'Analysis Summaries' or 'Recent News' to supplement your answer *if they provide relevant information not covered in the Retrieved Context*.
        5.  **Crucially, cite sources accurately.** When using information from the 'Retrieved Context', use the 'Source: <URL>' provided within that context block. Cite relevant sentences or paragraphs like this: "(Source: <URL>)".
        6.  If the retrieved context is insufficient, contradictory, or retrieval failed, clearly state this limitation in your report.
        7.  Do NOT make up information. Stick to the facts presented in the provided materials.
        8.  Format the report clearly using Markdown headings, lists, and bold text where appropriate.
        """)
    ]
    # --- End Synthesis Prompt ---

    try:
        print("--- Synthesis Node: Calling Synthesis LLM ---")
        response = synthesis_llm.invoke(synthesis_prompt_messages)
        print("--- Synthesis Node: LLM Response Received ---")
        # Return dictionary to update the final_report field
        return {"final_report": response.content}
    except Exception as e:
        error_msg = f"Error during final synthesis: {e}"; print(f"!!!!!!!! ERROR during synthesis: {error_msg} !!!!!!!!")
        traceback.print_exc(); current_errors = state.get("error_log", [])
        return {"final_report": f"Error: Failed to generate final report. {error_msg}", "error_log": current_errors + [error_msg]}


# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node) # Use standard ToolNode
workflow.add_node("retriever", retrieve_context_node)
workflow.add_node("synthesizer", synthesis_node)

workflow.set_entry_point("agent")

# Routing function - determines path after agent node makes a decision
def router(state: AgentState) -> str:
    print("--- Router: Deciding next step ---")
    messages = state["messages"]
    if not messages: print("--- Router: No messages found, ending ---"); return END
    last_message = messages[-1]

    # If agent generated tool calls, execute them
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print(f"--- Router: Tool calls requested -> 'action'")
        return "action"

    # If the last step was tool execution, always go back to the agent node
    # The agent node will then decide if processing is needed or call LLM
    if isinstance(last_message, ToolMessage):
        print(f"--- Router: Tool '{last_message.name}' finished -> 'agent'")
        return "agent"

    # If the last message is from the Agent and has NO tool calls
    # This means the agent has finished its planning/tool use phase
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
         if "Error: Could not invoke LLM" in last_message.content: print(f"--- Router: LLM error detected -> END"); return END
         else:
              # Proceed to retrieve context for final synthesis
              print(f"--- Router: Agent finished planning/tool use -> 'retriever'")
              return "retriever"

    # Fallback / safety end condition
    print(f"--- Router: Fallback condition (last message type: {type(last_message)}) -> END")
    return END

# Set up edges using the router
workflow.add_conditional_edges("agent", router, {"action": "action", "retriever": "retriever", END: END})
workflow.add_edge("action", "agent") # Route back to agent after ANY tool action
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile the graph
app = workflow.compile()
print("--- LangGraph Compiled (RAG Workflow Fully Integrated) ---")

# --- Runner Function (Simplified Return Block from #55) ---
async def run_agent_graph(query: str) -> dict:
    """Invokes the LangGraph agent with the user query using ainvoke."""
    print(f"\n--- Running Aether Analyst v2 for Query: '{query}' ---")
    initial_state = AgentState(messages=[HumanMessage(content=query)], query=query, scraped_data={}, processed_urls=set(), analysis_results={}, news_results=None, retrieved_context=None, final_report=None, error_log=[])
    final_state: Optional[Dict] = None
    final_report_str: str = "Error: Agent failed before synthesis." # Initial default

    try:
        print("--- Invoking Agent Graph using ainvoke... ---")
        final_state = await app.ainvoke(initial_state, {"recursion_limit": 35}) # Slightly increased limit again
        print("--- Agent Graph invocation finished ---")
    except Exception as e:
        error_msg = f"Error during graph invocation: {e}"; print(f"\n!!!!!!!! ERROR during graph invocation: {error_msg} !!!!!!!!")
        traceback.print_exc(); error_log = initial_state.get("error_log", []) + [f"Graph Invocation Error: {e}"]
        return {"markdown": f"Error during agent execution: {e}", "data": {"messages": initial_state.get("messages", []), "error_log": error_log}}

    # --- Process final_state (Simplified Version - Reverted final block logic from #55) ---
    print("--- Processing Final State ---")
    report_content = "Error: Could not determine final report content."
    data_to_return = {"error": "Final state was invalid or None"}

    if final_state and isinstance(final_state, dict):
        data_to_return = final_state # Good state
        report_from_state = final_state.get("final_report")
        if report_from_state and isinstance(report_from_state, str):
            report_content = report_from_state
            # print(f"DEBUG: Found final report content in state['final_report'].") # Removed Debug Print
        elif final_state.get("messages"):
            messages = final_state.get("messages", [])
            found_ai_message = False
            for msg in reversed(messages):
                 if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                      report_content = msg.content
                      found_ai_message = True
                      # print(f"DEBUG: Using last AIMessage content as fallback report.") # Removed Debug Print
                      break
            if not found_ai_message:
                report_content = "Agent finished. Last message was not an AI response with content."
                # print(f"DEBUG: No suitable final AI message found in state messages.") # Removed Debug Print
        else:
             report_content = "Agent finished without messages in final state."
             # print(f"DEBUG: No messages found in final state.") # Removed Debug Print

        # Append errors logged during the run
        error_log = final_state.get("error_log")
        if error_log and isinstance(error_log, list):
            if not isinstance(report_content, str):
                 report_content = json.dumps(report_content) # Ensure report_content is string before appending
            try:
                # Ensure all items in error_log are strings before joining
                error_string = "\n".join(map(str, error_log))
                report_content += "\n\n**Errors Encountered During Run:**\n" + error_string
            except TypeError as te:
                 print(f"ERROR appending error log: Items not strings? {te}")
                 report_content += "\n\n**Errors Encountered During Run:**\n[Could not format error log]"
    else:
        # print(f"DEBUG: Final state was None or not a dictionary: {final_state}") # Removed Debug Print
        report_content = f"Error: Agent run failed to produce a valid final state dictionary."


    # print(f"DEBUG: Returning markdown content snippet: {str(report_content)[:100]}...") # Removed Debug Print
    # Construct final return dictionary simply
    return {"markdown": str(report_content), "data": data_to_return}