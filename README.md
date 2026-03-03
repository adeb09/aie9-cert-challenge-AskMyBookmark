# AskMyBookmark — AIE9 Certification Challenge

Query your GitHub starred repositories using natural language, powered by RAG.

---

## Running Locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd aie9-cert-challenge-AskMyBookmark
```

### 2. Set up your environment variables

Create a `notebooks/.env` file with your OpenAI API key:

```bash
cp notebooks/.env.example notebooks/.env   # if the example exists, otherwise create it
```

Add the following to `notebooks/.env`:

```
OPENAI_API_KEY=sk-...
```

> `notebooks/.env` is listed in `.gitignore` and will never be committed.

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Start the app

```bash
uv run ./start-dev.sh
```

This starts both servers:
- **Backend** → http://localhost:8000 (FastAPI)
- **Frontend** → http://localhost:3000 (Next.js)

`npm install` for the frontend runs automatically on first launch.

### 5. Open the app

Go to **http://localhost:3000** in your browser.

Enter your GitHub Personal Access Token and click **Load My Bookmarks**.

> If `data/cached/github_data.pkl` exists, the app loads from that cache instead of calling the GitHub API — startup is much faster.

---

## Project Structure

```
app/
  ask_my_bookmark.py   # FastAPI backend — RAG pipeline + API endpoints
frontend/
  pages/index.tsx      # Next.js single-page UI
  styles/globals.css
data/
  cached/
    github_data.pkl    # Cached GitHub starred repo data (not committed)
notebooks/
  .env                 # Your API keys (not committed — add OPENAI_API_KEY here)
  POC_AskMyBookmark_cleaner.ipynb   # RAG pipeline prototyping notebook
```

---

## Stack

| Layer | Choice |
|---|---|
| LLM | `gpt-4.1-nano` (OpenAI) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Orchestrator | LangGraph |
| Vector DB | Qdrant (in-memory) |
| Backend | FastAPI + uvicorn |
| Frontend | Next.js (TypeScript) |

---

## Certification Challenge Sections

### 1) Problem & Audience
Describe the core problem (1 sentence) and why it matters to a specific audience (1–2 paragraphs).

### 2) Proposed Solution & Stack
Summarize UX flow and justify one tool per a16z stack layer:
LLM, Embeddings, Orchestrator, Vector DB, Eval, Validators, UI.

### 3) Data & Chunking
List data sources/APIs and chunking strategy (size + overlap + rationale).

### 4) End-to-End Prototype
Explain how the prototype runs locally or via API; include LangGraph reference.

### 5) Golden Test Set & RAGAS
Summarize test set, metrics table (Faithfulness, Response Relevancy, Context Precision, Context Recall),
and 1 strength + 1 improvement.

### 6) Advanced Retrieval
List retrieval techniques tried (BM25, Multi-Query, etc.) and their comparative outcomes.

### 7) Performance & Next Steps
Compare naive vs. advanced retrieval and outline 2–3 concrete next steps.

---

> Deliverables: see [`/deliverables/`](deliverables/) for the generated checklist and slide outline.
