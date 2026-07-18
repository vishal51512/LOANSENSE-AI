# LoanSense AI

LoanSense AI is a loan-assistant project that combines a FastAPI backend, Streamlit frontend, and a RAG pipeline over PDF loan documents.

## Project Walkthrough

### Core flow
1. Loan PDFs are stored in `data/` (default docs) or uploaded from UI into `uploads/`.
2. Documents are chunked and indexed in `vector_store/` using embeddings + BM25.
3. User questions go to `/chat`, are routed by the orchestrator, and answered using retrieval/interest/EMI agents.
4. Streamlit UI pages consume backend APIs and display answers, confidence, and source pages.

### Main folders
- `api/` - FastAPI app and routes (`/chat`, `/upload`, `/stats`, `/knowledge-base`, `/documents`)
- `frontend/` - Streamlit app and pages (Chat, Upload, Knowledge Base, Analytics)
- `agents/` - query orchestration and domain agents
- `rag/` - PDF processing, chunking, embedding, retrieval, index build
- `llm/` - LLM client, prompt building, response validation/formatting
- `tests/` - pytest tests

## Setup

### 1) Create virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add API key
Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The backend loads this variable in `llm/client.py`.

### 3) Set model
Default model is set in `llm/client.py`:

```python
self.model = "llama-3.3-70b-versatile"
```

To use another model, replace this value with a valid Groq model id.

## Run the Project

Open **two terminals** from project root.

### Terminal 1: Start API server
```bash
uvicorn api.server:app --reload
```
Backend runs on `http://127.0.0.1:8000`.

### Terminal 2: Start Streamlit frontend
```bash
streamlit run frontend/app.py
```
Frontend opens in browser (usually `http://localhost:8501`).

## Optional: Rebuild knowledge index from PDFs
If you add/replace PDFs directly in `data/` or `uploads/`, rebuild indexes:

```bash
python -m rag.build_index
```

## Run tests
```bash
python -m pytest
```
