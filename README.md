# 🛡️ Sentinel — AI Research Assistant

Sentinel is an AI-powered research assistant designed to provide reliable, context-aware answers from uploaded documents and web sources. It combines **LangGraph, CRAG-style corrective retrieval, Self-RAG-inspired answer verification, and Human-in-the-Loop (HITL) review** to reduce hallucinations and improve answer reliability.

---

## ✨ Features

- 📄 **Document Ingestion** — Upload PDF and TXT documents directly through the Streamlit interface.
- 🔎 **Document Retrieval** — Searches relevant chunks from uploaded documents instead of passing the entire document to the LLM.
- 🌐 **Web Search Fallback** — Uses web search when local document context is insufficient or irrelevant.
- 🧠 **LangGraph Agent Workflow** — Manages routing, planning, retrieval, generation, criticism, and human review.
- 🔄 **CRAG-style Corrective Retrieval** — Evaluates whether retrieved context is relevant and triggers web search when necessary.
- ✅ **Self-RAG-inspired Verification** — Evaluates generated responses for grounding and unsupported claims.
- 👤 **Human-in-the-Loop** — Allows users to review, edit, approve, or reject generated answers.
- 💬 **Conversation Memory** — Maintains conversation context across messages and sessions.
- 📚 **Source-Aware Responses** — Keeps track of retrieved evidence used to generate answers.
- ⚡ **FastAPI Backend** — Separates the agent backend from the Streamlit frontend.
- 💻 **Streamlit Frontend** — Provides an interactive chat interface with document upload and review controls.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Streamlit Frontend│
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  FastAPI Backend │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  LangGraph Agent │
                         └────────┬─────────┘
                                  │
                      ┌───────────▼───────────┐
                      │         Router        │
                      └───────────┬───────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   ▼                             ▼
             ┌───────────┐                ┌───────────┐
             │ Chitchat  │                │ Research  │
             └───────────┘                └─────┬─────┘
                                               │
                                               ▼
                                        ┌────────────┐
                                        │   Planner  │
                                        └─────┬──────┘
                                              │
                                              ▼
                                        ┌────────────┐
                                        │ Retriever  │
                                        └─────┬──────┘
                                              │
                         ┌────────────────────┴───────────────────┐
                         │                                        │
                         ▼                                        ▼
                 ┌────────────────┐                      ┌────────────────┐
                 │ Uploaded Docs  │                      │   Web Search   │
                 └───────┬────────┘                      └───────┬────────┘
                         │                                       │
                         └────────────────┬──────────────────────┘
                                          │
                                          ▼
                                  ┌─────────────────┐
                                  │  CRAG Grader    │
                                  └────────┬────────┘
                                           │
                                           ▼
                                    ┌────────────┐
                                    │ Generator  │
                                    └─────┬──────┘
                                          │
                                          ▼
                                    ┌────────────┐
                                    │   Critic   │
                                    └─────┬──────┘
                                          │
                             ┌────────────┴────────────┐
                             │                         │
                           Reject                    Pass
                             │                         │
                             ▼                         ▼
                        Re-retrieve                Human Review
                                                       │
                                             ┌─────────┴─────────┐
                                             │                   │
                                          Reject              Approve
                                             │                   │
                                             ▼                   ▼
                                        Re-run Agent         Save Answer

```





## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/saarthaktiwari-04710/Sentinel-Research-Assistant.git
cd Sentinel-Research-Assistant
```

### 2. Create a virtual environment

Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```
Linux/macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies

``` bash
pip install -r requirements.txt
```


### 4. Configure environment variables

Create a .env file in the project root:

``` bash
GROQ_API_KEY=your_groq_api_key_here
BACKEND_URL=http://localhost:8000
```

⚠️ Never commit your .env file or expose your API key publicly.

## ▶️ Run Sentinel

### Start the backend

Open a terminal and run:

``` bash
python backend.py
```
The FastAPI backend will run at:
``` bash
http://localhost:8000
```
### Start the frontend

Open another terminal and run:
``` bash
streamlit run frontend.py
```

Open the URL shown by Streamlit, usually:
``` bash
http://localhost:8501
```
