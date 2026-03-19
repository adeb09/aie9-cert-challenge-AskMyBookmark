# AskMyBookmark

Conversational search over your starred GitHub repositories.

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd aie9-cert-challenge-AskMyBookmark
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Start the app

```bash
uv run ./start-dev.sh
```

This starts both servers and opens your browser automatically:

- **Backend** → http://localhost:8000 (FastAPI)
- **Frontend** → http://localhost:3000 (Next.js)

Frontend dependencies (`npm install`) run automatically on first launch.

### 5. Connect your GitHub account

Enter your **GitHub Personal Access Token** and click **Load My Bookmarks**.

- On first run the app fetches your starred repos from GitHub and builds the search index — this takes a few minutes.
- On subsequent runs the index is loaded from disk, so startup is near-instant.

> **If your GitHub stars have changed** and you want a full reload, click **"No, start fresh"** when prompted about the cache at startup. This wipes the query result cache and rebuilds the index from the latest GitHub data.
