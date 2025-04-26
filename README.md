# ✨ Aether Analyst v2 — Autonomous Web Research Agent

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)  
[![Framework](https://img.shields.io/badge/Framework-LangGraph%20%7C%20FastAPI%20%7C%20Streamlit-orange)](https://python.langchain.com/docs/langgraph/)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://railway.app)  

---

## 🚀 Overview

**Aether Analyst v2** is an autonomous, AI-powered web research agent capable of searching, scraping, analyzing, and synthesizing information from across the internet into concise, sourced reports.

Designed with modularity, scalability, and intelligence at its core, Aether Analyst leverages LangGraph, microservices, and retrieval-augmented generation (RAG) pipelines to deliver reliable and actionable research outputs.

---

## 🛠️ Key Features

- 🧠 **LangGraph Orchestration:** Modular, stateful execution flows for complex research tasks.  
- 🌐 **FastAPI Microservices:** Isolated services for search, scraping, analysis, news aggregation, and vector storage.  
- 🕸️ **Advanced Web Scraping:** Playwright-based dynamic page rendering with intelligent content extraction.  
- 🗃️ **Retrieval-Augmented Generation (RAG):** Contextual memory with Qdrant vector database and sentence-transformer embeddings.  
- 🤖 **LLM-Enhanced Reasoning:** Powered by OpenAI's GPT-4o models for synthesis, summarization, and decision making.  
- 🐳 **Fully Containerized Deployment:** Docker and Docker Compose support for effortless launch and scaling.  
- 🖥️ **Streamlit UI:** Intuitive web interface for submitting queries and reviewing reports.  

---

## 🧩 System Architecture

```mermaid
graph TD
  A[User Query] --> B[LangGraph Agent]
  B --> C{Decision Point}
  C --> D1[Search Service (SerpAPI)]
  C --> D2[Scrape Service (Playwright)]
  C --> D3[Analyze Service (LLM)]
  C --> D4[News Service (NewsAPI)]
  D2 --> E[Processing Service (Chunk & Embed)]
  E --> F[Qdrant Vector Store]
  F --> G[Retrieve Relevant Context]
  G --> H[Final Synthesis (GPT-4o)]
  H --> I[Streamlit Report Display]

---

## 🛠️ Technology Stack

| Layer          | Technology                                      |
|:--------------|:------------------------------------------------|
| **Language**  | Python 3.11                                     |
| **Orchestration** | LangGraph, LangChain                        |
| **LLMs**      | OpenAI GPT-4o                                    |
| **Web Framework** | FastAPI, Uvicorn                             |
| **Scraping**  | Playwright, BeautifulSoup4                      |
| **Vector DB** | Qdrant                                          |
| **Embeddings**| Sentence Transformers (`all-MiniLM-L6-v2`)       |
| **Deployment**| Docker, Docker Compose                           |
| **Frontend**  | Streamlit                                       |
| **APIs**      | OpenAI, SerpAPI, NewsAPI                         |

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.11+  
- Docker Desktop (or Docker Engine + Compose v2)  
- Git  
- API Keys for:
  - OpenAI  
  - SerpAPI  
  - NewsAPI  

---

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Koushik1161/aether-analyst-v2.git
   cd aether-analyst-v2
   ```

2. **Configure environment variables**  
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys:
   # OPENAI_API_KEY, SERPAPI_API_KEY, NEWSAPI_API_KEY
   ```

3. **Launch the application**  
   ```bash
   docker-compose up --build -d
   ```

4. **Access the Streamlit UI**  
   Open your browser at [http://localhost:8501](http://localhost:8501)

---

### To stop the application

```bash
docker-compose down
```

---

## 📈 Microservices Overview

| Service              | Purpose                                      |
|----------------------|----------------------------------------------|
| `webapp`             | Streamlit UI + LangGraph Orchestration       |
| `search_service`     | Perform web searches via SerpAPI             |
| `scrape_service`     | Extract dynamic website content              |
| `analyze_service`    | Analyze and summarize web data               |
| `news_service`       | Fetch the latest news articles               |
| `processing_service` | Chunk, embed, and store documents            |
| `qdrant`             | Vector DB for semantic search and retrieval  |

---

## 🛡️ License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

# 🌟 Why Aether Analyst?

Modern information gathering requires intelligence, autonomy, and precision.  
**Aether Analyst v2** is built for the future — a system that not only fetches information, but also understands and synthesizes it into actionable insights.

> _"Not just an agent. A research partner."_
