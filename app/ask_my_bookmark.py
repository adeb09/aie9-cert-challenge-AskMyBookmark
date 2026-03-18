import asyncio
import base64
import hashlib
import json
import os
import pickle
import re
import shutil
import threading
import uuid
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gidgethub import GitHubException
from gidgethub.aiohttp import GitHubAPI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from searcharray import SearchArray
from stamina import retry, retry_context
from textacy import preprocessing

from app.orchestrator import (
    OrchestratorState,
    _preprocess_text,
    _multi_match_search,
    build_orchestrator_graph,
)

# Load env vars — try notebooks/.env first (local dev), then root .env
for _env_path in ["notebooks/.env", ".env"]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="AskMyBookmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global in-memory pipeline + session state (single-user local demo)
# ---------------------------------------------------------------------------

pipeline_state: Dict[str, Any] = {
    "status":          "idle",   # idle | loading | ready | error
    "phase":           None,     # fetching | indexing
    "github_username": None,     # resolved from the PAT via /user
    # fetching sub-steps: "discovering" | "fetching_docs"
    "fetch_step":      None,
    "repo_count":      0,
    "total_repos":     0,
    # indexing sub-steps: "loading_cache" | "bm25" | "embedding" | "compiling"
    "index_step":      None,
    "index_count":     0,        # repos embedded so far
    "index_total":     0,        # total repos to embed
    "orchestrator":    None,     # compiled LangGraph graph
    "error":           None,
}

# MemorySaver is long-lived; sessions are keyed by thread_id inside it
_checkpointer = MemorySaver()

# Active sessions: session_id → {"config": {"configurable": {"thread_id": ...}}}
# The actual graph state lives in _checkpointer; we only track configs here.
_sessions: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Text pre-processing (mirrors the notebook)
# ---------------------------------------------------------------------------

MAX_CHARACTERS = 30_000


def strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def make_normalize_text_pipeline(*, unicode_form: str = "NFC"):
    return preprocessing.make_pipeline(
        strip_markdown,
        preprocessing.remove.html_tags,
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.quotation_marks,
        partial(preprocessing.normalize.unicode, form=unicode_form),
        preprocessing.normalize.whitespace,
    )


_normalize_text = make_normalize_text_pipeline()
NAMESPACE = uuid.NAMESPACE_URL


def repo_to_uuid(repo_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, repo_name))


def _normalize_docs(docs: List[Dict]) -> str:
    content = "\n\n".join(d.get("content", "") for d in docs)
    if len(content) > MAX_CHARACTERS:
        content = content[:MAX_CHARACTERS]
    return _normalize_text(content)

# ---------------------------------------------------------------------------
# GitHub fetching helpers
# ---------------------------------------------------------------------------


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, GitHubException):
        return exc.status_code not in (404, 403)
    return isinstance(exc, aiohttp.ClientError)


@retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def _fetch_markdown_content(gh, owner, repo, file_path):
    try:
        data = await gh.getitem(f"/repos/{owner}/{repo}/contents/{file_path}")
        return {
            "name": data["name"], "path": data["path"], "size": data["size"],
            "content": base64.b64decode(data["content"]).decode("utf-8"),
            "success": True,
        }
    except GitHubException as e:
        return {"path": file_path, "success": False, "error": str(e)}


@retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def _get_root_markdown_files(gh, owner, repo):
    try:
        contents = await gh.getitem(f"/repos/{owner}/{repo}/contents/")
        return [f for f in contents if f["type"] == "file" and f["name"].lower().endswith(".md")]
    except GitHubException:
        return []


async def fetch_starred_repos_with_docs(
    github_token: str,
    max_repos: Optional[int] = None,
    concurrent_tasks: int = 20,
) -> List[Dict]:
    semaphore = asyncio.Semaphore(concurrent_tasks)
    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "ask-my-bookmark", oauth_token=github_token)

        # Phase 1 — paginate through starred repos (total unknown until done)
        pipeline_state["fetch_step"] = "discovering"
        starred: List[Dict] = []
        async for repo in gh.getiter("/user/starred", accept="application/vnd.github.mercy-preview+json"):
            starred.append(repo)
            pipeline_state["total_repos"] = len(starred)   # live count as pages arrive
            if max_repos and len(starred) >= max_repos:
                break

        # Phase 2 — fetch READMEs / markdown docs for each repo
        pipeline_state["fetch_step"] = "fetching_docs"
        pipeline_state["repo_count"] = 0

        async def fetch_repo_docs(repo: Dict) -> Dict:
            owner     = repo["owner"]["login"]
            name      = repo["name"]
            full_name = repo["full_name"]
            topics    = repo.get("topics") or []
            base = {
                "repo":        full_name,
                "description": repo.get("description"),
                "topics":      topics,
                "stars":       repo.get("stargazers_count"),
                "language":    repo.get("language"),
                "url":         repo.get("html_url"),
            }
            try:
                async for attempt in retry_context(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0):
                    with attempt:
                        readme_data = await gh.getitem(f"/repos/{owner}/{name}/readme")
                content = base64.b64decode(readme_data["content"]).decode("utf-8")
                return {**base, "doc_source": "readme", "docs": [{"name": readme_data["name"], "path": readme_data["path"], "size": readme_data["size"], "content": content}]}
            except GitHubException as e:
                if e.status_code != 404:
                    print(f"Warning: unexpected error fetching README for {full_name}: {e}")
            md_files = await _get_root_markdown_files(gh, owner, name)
            if md_files:
                file_results = await asyncio.gather(*[_fetch_markdown_content(gh, owner, name, f["path"]) for f in md_files])
                return {**base, "doc_source": "root_markdown", "docs": [r for r in file_results if r.get("success")]}
            return {**base, "doc_source": None, "docs": []}

        async def fetch_throttled(repo: Dict) -> Dict:
            async with semaphore:
                result = await fetch_repo_docs(repo)
                pipeline_state["repo_count"] += 1
                return result

        return await asyncio.gather(*[fetch_throttled(repo) for repo in starred])

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cached")

CURATED_QUERY = "awesome curated lists"


# ---------------------------------------------------------------------------
# Per-user cache path resolution
# ---------------------------------------------------------------------------

async def _resolve_github_username(github_token: str) -> str:
    """Return the GitHub login for *github_token*, falling back to a short token
    hash if the /user endpoint is unreachable or the token lacks that scope."""
    try:
        async with aiohttp.ClientSession() as session:
            gh = GitHubAPI(session, "ask-my-bookmark", oauth_token=github_token)
            user = await gh.getitem("/user")
            login = str(user.get("login") or "").strip()
            if login:
                # Sanitise: keep only safe filesystem characters
                return re.sub(r"[^a-zA-Z0-9_\-]", "_", login).lower()
    except Exception:
        pass
    return "user_" + hashlib.md5(github_token.encode()).hexdigest()[:8]


def _make_cache_paths(username: str) -> Dict[str, str]:
    """Return a dict of all cache file/dir paths namespaced under *username*."""
    user_dir = os.path.join(_CACHE_DIR, username)
    return {
        "dir":         user_dir,
        "github_data": os.path.join(user_dir, "github_data.pkl"),
        "search_df":   os.path.join(user_dir, "search_df.pkl"),
        "qdrant":      os.path.join(user_dir, "qdrant_store"),
        "index_meta":  os.path.join(user_dir, "index_meta.json"),
    }


# ---------------------------------------------------------------------------
# Index-cache helpers
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    """Return the MD5 hex digest of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_index_cache_valid(paths: Dict[str, str]) -> bool:
    """Return True when search_df + qdrant_store caches exist and were built
    from the same github_data.pkl that is currently on disk."""
    for p in (paths["github_data"], paths["search_df"], paths["qdrant"], paths["index_meta"]):
        if not os.path.exists(p):
            return False
    try:
        with open(paths["index_meta"]) as f:
            meta = json.load(f)
        return meta.get("github_data_hash") == _hash_file(paths["github_data"])
    except Exception:
        return False


def _save_index_meta(paths: Dict[str, str]) -> None:
    """Record the hash of github_data.pkl so we can detect staleness later."""
    with open(paths["index_meta"], "w") as f:
        json.dump({"github_data_hash": _hash_file(paths["github_data"])}, f)


def _build_search_df(repo_data: List[Dict]) -> pd.DataFrame:
    """Build the per-field search DataFrame used by MultiMatchBM25Retriever."""
    rows = []
    for repo in repo_data:
        rid    = repo_to_uuid(repo["repo"])
        topics = repo.get("topics") or []
        docs   = repo.get("docs") or []
        rows.append({
            "id":          rid,
            "repo":        repo["repo"],
            "description": repo.get("description") or "",
            "topics":      " ".join(topics),
            "topics_list": topics,
            "language":    repo.get("language"),
            "stars":       repo.get("stars"),
            "url":         repo.get("url"),
            "content":     f"Topics: {','.join(topics)}\n" + _normalize_docs(docs),
        })
    df = pd.DataFrame(rows)

    # Index fields with SearchArray for BM25 scoring
    df["repo_idx"]        = SearchArray.index(df["repo"],        tokenizer=_preprocess_text)
    df["description_idx"] = SearchArray.index(df["description"], tokenizer=_preprocess_text)
    df["topics_idx"]      = SearchArray.index(df["topics"],      tokenizer=_preprocess_text)
    df["content_idx"]     = SearchArray.index(df["content"],     tokenizer=_preprocess_text)

    return df


def _compute_curated_scores(search_df: pd.DataFrame) -> None:
    """Add curated_list_bm25 column to search_df in-place."""
    curated_results = _multi_match_search(
        query=CURATED_QUERY,
        df=search_df,
        columns=["repo_idx", "topics_idx", "description_idx", "content_idx"],
        boosts={"repo_idx": 3.0, "topics_idx": 2.0, "description_idx": 1.5, "content_idx": 1.0},
    )
    score_by_id = dict(zip(curated_results["id"], curated_results["score"].astype(float)))
    search_df["curated_list_bm25"] = search_df["id"].map(score_by_id).fillna(0.0)


_EMBED_BATCH_SIZE = 50  # repos per add_texts call — controls progress granularity


def _build_vector_store(search_df: pd.DataFrame, cache_path: str) -> QdrantVectorStore:
    """Embed all repo content and persist the Qdrant collection to *cache_path*.
    Updates pipeline_state['index_count'] after every batch so the frontend can
    show a real progress bar."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)
    client = QdrantClient(path=cache_path)
    client.create_collection(
        collection_name="ask_my_bookmark",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vs = QdrantVectorStore(client=client, collection_name="ask_my_bookmark", embedding=embeddings)

    texts     = search_df["content"].tolist()
    ids       = search_df["id"].tolist()
    metadatas = (
        search_df[["id", "repo", "description", "topics_list", "language", "stars", "url", "curated_list_bm25"]]
        .rename(columns={"topics_list": "topics"})
        .to_dict("records")
    )

    pipeline_state["index_total"] = len(texts)
    pipeline_state["index_count"] = 0

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        end = start + _EMBED_BATCH_SIZE
        vs.add_texts(
            texts=texts[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        pipeline_state["index_count"] = min(end, len(texts))

    return vs


def _load_vector_store(cache_path: str) -> QdrantVectorStore:
    """Reconnect to an existing on-disk Qdrant collection — no re-embedding."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(path=cache_path)
    return QdrantVectorStore(client=client, collection_name="ask_my_bookmark", embedding=embeddings)

# ---------------------------------------------------------------------------
# SSE streaming helpers for query progress
# ---------------------------------------------------------------------------

# Maps LangGraph node names → user-visible progress labels and step numbers.
# Only nodes worth showing (i.e. ones with meaningful compute time) are listed.
NODE_PROGRESS: Dict[str, Dict] = {
    "query_prep":     {"label": "Analyzing your query",   "step": 1},
    "refine_query":   {"label": "Refining your query",    "step": 1},
    "lexical_search": {"label": "Searching repositories", "step": 2},
    "ensemble_search":{"label": "Searching repositories", "step": 2},
    "merge_results":  {"label": "Merging results",        "step": 3},
    "filter_results": {"label": "Filtering results",      "step": 4},
    "rerank_results": {"label": "Ranking by relevance",   "step": 5},
    "generate_answer":{"label": "Generating answer",      "step": 6},
}
TOTAL_QUERY_STEPS = 6


def _sse(data: dict) -> str:
    """Encode a dict as a single Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_graph(
    orchestrator,
    graph_input: Any,
    config: dict,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Run the LangGraph graph in a background thread, yielding SSE strings for
    each tracked node that completes, then a final 'result' event."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _run() -> None:
        try:
            for chunk in orchestrator.stream(graph_input, config, stream_mode="updates"):
                for node_name in chunk:
                    if node_name in NODE_PROGRESS:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type":        "progress",
                                "label":       NODE_PROGRESS[node_name]["label"],
                                "step":        NODE_PROGRESS[node_name]["step"],
                                "total_steps": TOTAL_QUERY_STEPS,
                            },
                        )
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "error": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    error_occurred = False
    while True:
        item = await queue.get()
        if item is None:
            break
        yield _sse(item)
        if item.get("type") == "error":
            error_occurred = True

    if error_occurred:
        return

    # Graph paused at human_feedback interrupt (or reached END); surface final state.
    snap  = await asyncio.to_thread(orchestrator.get_state, config)
    done  = not bool(snap.next)
    if done:
        _sessions.pop(session_id, None)
    yield _sse({"type": "result", **_state_to_response(snap.values, session_id, done)})


# ---------------------------------------------------------------------------
# Background pipeline build
# ---------------------------------------------------------------------------


async def _build_pipeline(github_token: str) -> None:
    try:
        pipeline_state.update({
            "status": "loading", "phase": "fetching",
            "github_username": None,
            "fetch_step": "discovering",
            "repo_count": 0, "total_repos": 0,
            "index_step": None, "index_count": 0, "index_total": 0,
            "orchestrator": None, "error": None,
        })
        os.makedirs(_CACHE_DIR, exist_ok=True)

        # ── Resolve GitHub username → per-user cache paths ───────────────────
        username = await _resolve_github_username(github_token)
        pipeline_state["github_username"] = username
        paths = _make_cache_paths(username)
        os.makedirs(paths["dir"], exist_ok=True)
        print(f"GitHub username resolved: {username} (cache dir: {paths['dir']})")

        # ── Step 1: raw GitHub data ──────────────────────────────────────────
        if os.path.exists(paths["github_data"]):
            print(f"GitHub data cache found for {username} — skipping API fetch.")
            pipeline_state["phase"] = "indexing"

            def _load_github():
                with open(paths["github_data"], "rb") as f:
                    return pickle.load(f)

            repo_data: List[Dict] = await asyncio.to_thread(_load_github)
            pipeline_state["total_repos"] = len(repo_data)
            pipeline_state["repo_count"]  = len(repo_data)
        else:
            print(f"No GitHub data cache for {username} — fetching starred repos.")
            repo_data = await fetch_starred_repos_with_docs(github_token)
            def _save_github():
                with open(paths["github_data"], "wb") as f:
                    pickle.dump(repo_data, f)
            await asyncio.to_thread(_save_github)
            print(f"GitHub data cached at {paths['github_data']}")

        pipeline_state["phase"] = "indexing"

        # ── Step 2: BM25 + vector indices (smart cache) ──────────────────────
        if _is_index_cache_valid(paths):
            print(f"Index cache valid for {username} — loading from disk (no re-embedding).")
            pipeline_state["index_step"] = "loading_cache"

            def _load_indices():
                with open(paths["search_df"], "rb") as f:
                    df = pickle.load(f)
                vs = _load_vector_store(paths["qdrant"])
                return df, vs

            search_df, vector_store = await asyncio.to_thread(_load_indices)
            print(f"Loaded {len(search_df)} rows from search_df cache.")

        else:
            print(f"Index cache missing or stale for {username} — rebuilding.")

            pipeline_state["index_step"] = "bm25"

            def _build_indices():
                df = _build_search_df(repo_data)
                _compute_curated_scores(df)
                return df

            search_df = await asyncio.to_thread(_build_indices)
            print(f"search_df built: {len(search_df)} rows")

            pipeline_state["index_step"] = "embedding"
            vector_store = await asyncio.to_thread(_build_vector_store, search_df, paths["qdrant"])
            print("Vector store built and persisted to disk.")

            def _save_indices():
                with open(paths["search_df"], "wb") as f:
                    pickle.dump(search_df, f)
                _save_index_meta(paths)

            await asyncio.to_thread(_save_indices)
            print(f"Index cache saved for {username}.")

        # ── Step 3: compile orchestrator graph ───────────────────────────────
        pipeline_state["index_step"] = "compiling"

        def _compile():
            return build_orchestrator_graph(
                search_df=search_df,
                vector_store=vector_store,
                checkpointer=_checkpointer,
            )

        orchestrator = await asyncio.to_thread(_compile)
        print("Orchestrator graph compiled.")

        pipeline_state.update({
            "orchestrator": orchestrator,
            "status":       "ready",
            "phase":        None,
            "index_step":   None,
        })
        print(f"Pipeline ready — {pipeline_state['total_repos']} repos indexed for @{username}.")

    except Exception as exc:
        pipeline_state.update({"status": "error", "phase": None, "error": str(exc)})
        print(f"Pipeline build error: {exc}")

# ---------------------------------------------------------------------------
# Shared helpers for session endpoints
# ---------------------------------------------------------------------------

def _make_initial_state(query: str, top_k: int) -> OrchestratorState:
    return {
        "query":              query,
        "keywords":           [],
        "expansions":         [],
        "bm25_terms":         [],
        "route":              "",
        "include_curated":    False,
        "bm25_results":       [],
        "vector_results":     [],
        "merged_results":     [],
        "top_k":              top_k,
        "answer":             "",
        "feedback":           {},
        "blocklist":          [],
        "good_repos":         [],
        "feedback_iteration": 0,
    }


def _state_to_response(state: OrchestratorState, session_id: str, done: bool) -> dict:
    """Convert graph state to a JSON-serialisable session response."""
    results = []
    for doc in (state.get("merged_results") or []):
        m = doc.metadata
        repo = m.get("repo", "unknown")
        results.append({
            "repo":        repo,
            "url":         m.get("url") or f"https://github.com/{repo}",
            "description": m.get("description") or "",
            "language":    m.get("language") or "",
            "stars":       m.get("stars"),
            "topics":      m.get("topics") or [],
        })
    return {
        "session_id": session_id,
        "answer":     state.get("answer") or "",
        "results":    results,
        "iteration":  state.get("feedback_iteration") or 0,
        "done":       done,
    }

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SetupRequest(BaseModel):
    github_token: str


class SessionStartRequest(BaseModel):
    question: str
    top_k: int = 10


class SessionFeedbackRequest(BaseModel):
    session_id: str
    ratings:    Dict[str, str]   # repo → "good"|"meh"|"bad"
    done:       bool = False     # True = user clicked "Done"


# Kept for backward-compat with any existing clients
class QueryRequest(BaseModel):
    question: str

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/setup")
async def setup(request: SetupRequest):
    if pipeline_state["status"] == "loading":
        return {"status": "loading", "message": "Pipeline is already being built."}
    asyncio.create_task(_build_pipeline(request.github_token))
    return {"status": "loading", "message": "Pipeline build started."}


@app.get("/api/status")
async def status():
    return {
        "status":          pipeline_state["status"],
        "phase":           pipeline_state["phase"],
        "github_username": pipeline_state["github_username"],
        "fetch_step":      pipeline_state["fetch_step"],
        "repo_count":      pipeline_state["repo_count"],
        "total_repos":     pipeline_state["total_repos"],
        "index_step":      pipeline_state["index_step"],
        "index_count":     pipeline_state["index_count"],
        "index_total":     pipeline_state["index_total"],
        "error":           pipeline_state["error"],
    }


@app.post("/api/session/start")
async def session_start(request: SessionStartRequest):
    """Start a new search session, streaming SSE progress events then a final result."""
    if pipeline_state["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline not ready. Status: {pipeline_state['status']}",
        )

    orchestrator  = pipeline_state["orchestrator"]
    session_id    = str(uuid.uuid4())
    config        = {"configurable": {"thread_id": session_id}}
    _sessions[session_id] = {"config": config}

    initial_state = _make_initial_state(request.question, request.top_k)

    async def generate():
        # Send session_id immediately so the client can reference it for feedback
        yield _sse({"type": "session_created", "session_id": session_id})
        async for chunk in _stream_graph(orchestrator, initial_state, config, session_id):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session/feedback")
async def session_feedback(request: SessionFeedbackRequest):
    """Resume a session with feedback, streaming SSE progress events then a final result."""
    if pipeline_state["status"] != "ready":
        raise HTTPException(status_code=400, detail="Pipeline not ready.")

    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    orchestrator = pipeline_state["orchestrator"]
    config       = _sessions[request.session_id]["config"]

    ratings = dict(request.ratings)
    if request.done:
        ratings["__stop"] = True

    async def generate():
        async for chunk in _stream_graph(
            orchestrator, Command(resume=ratings), config, request.session_id
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query")
async def query(request: QueryRequest):
    """Simple stateless query endpoint (backward-compat). No feedback loop."""
    if pipeline_state["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline not ready. Status: {pipeline_state['status']}",
        )

    orchestrator = pipeline_state["orchestrator"]
    session_id   = str(uuid.uuid4())
    config       = {"configurable": {"thread_id": session_id}}
    initial_state = _make_initial_state(request.question, top_k=10)

    state = await asyncio.to_thread(orchestrator.invoke, initial_state, config)
    return {"response": state.get("answer", "")}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.ask_my_bookmark:app", host="0.0.0.0", port=8000, reload=True)
