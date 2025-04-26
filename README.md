# ✨ Aether Analyst v2: AI Web Research Agent

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LangGraph%20%7C%20FastAPI%20%7C%20Streamlit-orange)](https://python.langchain.com/docs/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
*(Optional: Add other relevant badges)*

## Overview

Aether Analyst is an autonomous AI agent designed to perform comprehensive web research based on user queries. It leverages a sophisticated architecture involving multiple microservices, advanced web scraping (Playwright), a Retrieval-Augmented Generation (RAG) pipeline with a vector database (Qdrant), and stateful orchestration (LangGraph) to gather, process, analyze, retrieve, and synthesize information from the web, culminating in a structured Markdown report with source citations.

This project was developed as part of the Masonry AI Agent Developer Hiring Assignment.

## Features

* **Advanced Orchestration:** Uses LangGraph to manage complex, multi-step agent workflows including planning (basic), tool execution, RAG data processing, context retrieval, and final synthesis.
* **Microservice Architecture:** Core tools (Search, Scrape, Analyze, News, RAG Processing) are implemented as independent FastAPI services for modularity and scalability.
* **Robust Web Scraping:** Utilizes Playwright via a dedicated service to handle dynamic websites and JavaScript rendering, providing more reliable content extraction than basic HTTP requests.
* **Retrieval-Augmented Generation (RAG):**
    * Chunks scraped web content.
    * Generates embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
    * Stores text chunks and embeddings in a Qdrant vector database.
    * Retrieves relevant chunks based on the user query prior to synthesis.
    * Synthesizes the final report grounded in the retrieved context with citations.
* **Multi-LLM Usage:** Employs OpenAI models (e.g., GPT-4o) for different tasks like orchestration/planning, content analysis (via service), and final synthesis.
* **Containerized:** Fully containerized using Docker and Docker Compose for easy local setup, reproducibility, and deployment readiness.
* **Web UI:** Simple user interface built with Streamlit.

## Architecture Overview

The system consists of several containerized services managed by Docker Compose:

1.  **`webapp` (Streamlit + LangGraph Orchestrator):**
    * Provides the user interface via Streamlit (`main.py`).
    * Hosts the LangGraph agent (`orchestrator/agent_graph.py`) which manages the overall workflow, calls tools, processes results, manages state, and interacts with the core LLM.
    * Communicates with Tool Services via HTTP REST APIs.
    * Communicates with Qdrant for retrieval.
2.  **Tool Services (FastAPI):**
    * `search_service`: Performs web searches (using SerpAPI).
    * `scrape_service`: Scrapes web pages using Playwright.
    * `analyze_service`: Analyzes content using OpenAI API.
    * `news_service`: Fetches news using NewsAPI.
    * `processing_service`: Chunks text, generates embeddings, stores in Qdrant.
3.  **`qdrant`:** Vector database instance storing processed web content.

**Data Flow (Simplified):**

User Query -> `webapp` (Streamlit UI) -> `webapp` (LangGraph Agent) -> `search_service` -> Agent -> `scrape_service` -> Agent -> `processing_service` -> Qdrant -> Agent -> `analyze_service` -> Agent -> `retrieve_context_node` (Qdrant) -> `synthesis_node` (LLM) -> Final Report -> `webapp` (Streamlit UI)

*[--> COMMENT: Create a flowchart image (e.g., using Mermaid in Markdown, draw.io, Excalidraw) illustrating this flow and embed it here or link to a separate ARCHITECTURE.md file containing the diagram and more detailed descriptions of the LangGraph nodes/edges and decision logic. <--]*

**(Link to Detailed Architecture Document/Diagram Here)**

## Tech Stack

* **Language:** Python 3.11+
* **Orchestration:** LangGraph (`langgraph`), LangChain (`langchain-core`, `langchain-openai`)
* **LLMs:** OpenAI GPT-4o (via `langchain-openai`)
* **Web Framework:** FastAPI, Uvicorn (for tool services)
* **Web Scraping:** Playwright, BeautifulSoup4
* **Vector DB:** Qdrant (`qdrant-client`)
* **Embeddings:** Sentence Transformers (`sentence-transformers`, `all-MiniLM-L6-v2` model)
* **External APIs:** SerpAPI, NewsAPI, OpenAI API
* **Containerization:** Docker, Docker Compose
* **UI:** Streamlit
* **Dependencies:** See `requirements.txt` for full list.

## Setup Instructions

**Prerequisites:**

* Python 3.11 or higher
* Docker Desktop (or Docker Engine + Compose V2) installed and running.
* Git
* API Keys:
    * OpenAI
    * SerpAPI
    * NewsAPI

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone [--> YOUR_REPOSITORY_URL_HERE <--]
    cd aether-analyst-v2
    ```
2.  **Set up Environment Variables:**
    * Rename the example environment file: `mv .env.example .env`
        *(Note: If you don't have `.env.example`, copy your existing `.env` to `.env.example` and REMOVE your secret keys from `.env.example` before committing).*
    * Edit the `.env` file and add your actual API keys for:
        * `OPENAI_API_KEY`
        * `SERPAPI_API_KEY`
        * `NEWSAPI_API_KEY`
    * Ensure `MCP_API_KEY` is set to a secure, unique string for internal authentication (the default `"aEth3r-An4lyst-L0cAl-K3y!7&p@9Z"` is fine for local use if quoted).
    * *(Optional)* Add your `LANGCHAIN_API_KEY` and set `LANGCHAIN_TRACING_V2="true"` if you wish to use LangSmith tracing.

## Running the Application (Docker Compose)

This is the standard way to run the full application stack locally.

1.  **Ensure Docker Desktop is running.**
2.  **Navigate to the project root directory (`aether-analyst-v2`) in your terminal.**
3.  **Build and start all services in the background:**
    ```bash
    docker-compose up --build -d
    ```
    *(The first build will take several minutes to download base images and install dependencies).*
4.  **Verify services are running:** Wait a minute or two for initialization (especially model loading), then check the status:
    ```bash
    docker-compose ps
    ```
    *You should see all 7 services (`qdrant_db`, `search_service`, `scrape_service`, `analyze_service`, `news_service`, `processing_service`, `webapp_ui`) listed with State/Status as `Up` or `Running`.*
5.  **View Logs (Optional):** To see the logs from all containers:
    ```bash
    docker-compose logs -f
    ```
    To see logs for a specific service (e.g., the webapp/orchestrator):
    ```bash
    docker-compose logs -f webapp
    ```
    (Press `Ctrl+C` to stop following logs).
6.  **Access the UI:** Open your web browser to `http://localhost:8501`.

**To stop the application:**

```bash
docker-compose down