import asyncio
import base64
import os
import pickle
import re
import uuid
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gidgethub import GitHubException
from gidgethub.aiohttp import GitHubAPI
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from stamina import retry, retry_context
from textacy import preprocessing
from typing import TypedDict

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
# Global in-memory pipeline state (single-user local demo)
# ---------------------------------------------------------------------------

pipeline_state: Dict[str, Any] = {
    "status": "idle",   # idle | loading | ready | error
    "phase": None,      # fetching | indexing  (only meaningful during loading)
    "repo_count": 0,    # repos fetched so far (updated during 'fetching' phase)
    "total_repos": 0,   # total starred repos discovered
    "rag_graph": None,
    "error": None,
}

# ---------------------------------------------------------------------------
# Text pre-processing (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

MAX_CHARACTERS = 30_000


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax; keep inner text and emojis."""
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


def organize_repo_data(
    repo_data: List[Dict],
) -> Tuple[List[str], List[Dict], List[str]]:
    ids: List[str] = []
    metadata: List[Dict] = []
    docs: List[str] = []

    for repo in repo_data:
        rid = repo_to_uuid(repo["repo"])
        topics: List[str] = repo.get("topics") or []

        ids.append(rid)
        metadata.append(
            {
                "id": rid,
                "repo": repo["repo"],
                "description": repo.get("description") or "",
                "topics": topics,
                "language": repo.get("language") or "",
                "doc_source": repo.get("doc_source"),
                "stars": repo.get("stars") or 0,
                "url": repo.get("url") or "",
            }
        )
        topics_str = f"Topics: {','.join(topics)}\n" if topics else ""
        docs.append(topics_str + _normalize_docs(repo.get("docs") or []))

    return ids, metadata, docs


# ---------------------------------------------------------------------------
# GitHub fetching helpers (mirrors the notebook exactly)
# ---------------------------------------------------------------------------


def _is_retriable(exc: Exception) -> bool:
    """Retry on transient network/server errors; skip deterministic 404/403."""
    if isinstance(exc, GitHubException):
        return exc.status_code not in (404, 403)
    return isinstance(exc, aiohttp.ClientError)


@retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def _fetch_markdown_content(
    gh: GitHubAPI, owner: str, repo: str, file_path: str
) -> Dict:
    try:
        data = await gh.getitem(f"/repos/{owner}/{repo}/contents/{file_path}")
        return {
            "name": data["name"],
            "path": data["path"],
            "size": data["size"],
            "content": base64.b64decode(data["content"]).decode("utf-8"),
            "success": True,
        }
    except GitHubException as e:
        return {"path": file_path, "success": False, "error": str(e)}


@retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def _get_root_markdown_files(
    gh: GitHubAPI, owner: str, repo: str
) -> List[Dict]:
    try:
        contents = await gh.getitem(f"/repos/{owner}/{repo}/contents/")
        return [
            f
            for f in contents
            if f["type"] == "file" and f["name"].lower().endswith(".md")
        ]
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

        # Step 1: collect full starred list (sequential; pagination handled by getiter)
        starred: List[Dict] = []
        async for repo in gh.getiter(
            "/user/starred",
            accept="application/vnd.github.mercy-preview+json",
        ):
            starred.append(repo)
            if max_repos and len(starred) >= max_repos:
                break

        pipeline_state["total_repos"] = len(starred)
        pipeline_state["repo_count"] = 0

        # Step 2: per-repo coroutine
        async def fetch_repo_docs(repo: Dict) -> Dict:
            owner = repo["owner"]["login"]
            name = repo["name"]
            full_name = repo["full_name"]
            topics: List[str] = repo.get("topics") or []

            base = {
                "repo": full_name,
                "description": repo.get("description"),
                "topics": topics,
                "stars": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "url": repo.get("html_url"),
            }

            # Try the dedicated /readme endpoint first
            try:
                async for attempt in retry_context(
                    on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0
                ):
                    with attempt:
                        readme_data = await gh.getitem(
                            f"/repos/{owner}/{name}/readme"
                        )
                content = base64.b64decode(readme_data["content"]).decode("utf-8")
                return {
                    **base,
                    "doc_source": "readme",
                    "docs": [
                        {
                            "name": readme_data["name"],
                            "path": readme_data["path"],
                            "size": readme_data["size"],
                            "content": content,
                        }
                    ],
                }
            except GitHubException as e:
                if e.status_code != 404:
                    print(
                        f"Warning: unexpected error fetching README for {full_name}: {e}"
                    )

            # Fallback: root-level .md files
            md_files = await _get_root_markdown_files(gh, owner, name)
            if md_files:
                file_tasks = [
                    _fetch_markdown_content(gh, owner, name, f["path"])
                    for f in md_files
                ]
                file_results = await asyncio.gather(*file_tasks)
                return {
                    **base,
                    "doc_source": "root_markdown",
                    "docs": [r for r in file_results if r.get("success")],
                }

            return {**base, "doc_source": None, "docs": []}

        # Step 3: throttled fan-out with live progress counter
        async def fetch_throttled(repo: Dict) -> Dict:
            async with semaphore:
                result = await fetch_repo_docs(repo)
                pipeline_state["repo_count"] += 1
                return result

        results: List[Dict] = await asyncio.gather(
            *[fetch_throttled(repo) for repo in starred]
        )

    return results


# ---------------------------------------------------------------------------
# RAG prompt & graph builder
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are AskMyBookmark, a personal research assistant with access to the user's GitHub starred repositories.

Your job is to help the user discover, recall, and explore repositories they have bookmarked on GitHub. You answer questions by reasoning over the retrieved repository context provided to you — not from your general knowledge of what exists on GitHub.

**Ground rules:**
- Only surface repositories that appear in the retrieved context below. Do not invent or suggest repositories that are not present in the context.
- If no retrieved repositories are relevant to the query, say so honestly and suggest the user try rephrasing or broadening their search.
- You may use your general knowledge to explain a topic or technology, but all repository recommendations must come exclusively from the retrieved context.

**When presenting results:**
- Always include the repository's full name (Repo) as a markdown link to its GitHub URL: [Repo](URL)
- Include a brief description of what the repo does (from the description and topics fields), written in your own words if the original description is terse or absent.
- Explain in 1–2 sentences *why* this repository is relevant to the user's query — this is the most important part.
- Group or rank results by relevance if there are several.
- If useful, note the primary programming language, star count, or topics to help the user evaluate the match.

**Tone:** Conversational, concise, and helpful. Treat the user as a developer who starred these repos intentionally and wants quick, intelligent recall — not a tutorial.

---

Retrieved repository context:
{context}"""


def _format_context(docs: List[Document]) -> str:
    chunks = []
    for doc in docs:
        meta = doc.metadata
        topics: List[str] = meta.get("topics") or []
        topics_str = f"Topics: {','.join(topics)}" if topics else "N/A"
        chunk = (
            f"---\n"
            f"Repo: {meta.get('repo', 'N/A')}\n"
            f"URL: {meta.get('url', 'N/A')}\n"
            f"Description: {meta.get('description', 'N/A')}\n"
            f"{topics_str}\n"
            f"Programming Language: {meta.get('language', 'N/A')}\n"
            f"Stars: {meta.get('stars', 'N/A')}\n\n"
            f"README excerpt:\n{doc.page_content.strip()}\n"
            f"---"
        )
        chunks.append(chunk)
    return "\n\n".join(chunks)


def build_rag_graph(retriever):
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4.1-nano")

    class State(TypedDict):
        question: str
        context: List[Document]
        response: str

    async def retrieve_node(state: State):
        docs = await retriever.ainvoke(state["question"])
        return {"context": docs}

    async def generate_node(state: State):
        context_str = _format_context(state["context"])
        messages = rag_prompt.format_messages(
            question=state["question"], context=context_str
        )
        response = await llm.ainvoke(messages)
        return {"response": response.content}

    builder = StateGraph(State).add_sequence([retrieve_node, generate_node])
    builder.add_edge(START, "retrieve_node")
    return builder.compile()


# ---------------------------------------------------------------------------
# Background pipeline build task
# ---------------------------------------------------------------------------


CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "cached", "github_data.pkl"
)


async def _build_pipeline(github_token: str) -> None:
    try:
        pipeline_state.update(
            {
                "status": "loading",
                "phase": "fetching",
                "repo_count": 0,
                "total_repos": 0,
                "rag_graph": None,
                "error": None,
            }
        )

        # 1. Load from local cache when available, otherwise hit GitHub API
        if os.path.exists(CACHE_PATH):
            print(f"Cache found at {CACHE_PATH} — skipping GitHub API fetch.")
            pipeline_state["phase"] = "indexing"

            def _load_cache():
                with open(CACHE_PATH, "rb") as f:
                    return pickle.load(f)

            repo_data: List[Dict] = await asyncio.to_thread(_load_cache)
            pipeline_state["total_repos"] = len(repo_data)
            pipeline_state["repo_count"] = len(repo_data)
        else:
            print("No cache found — fetching starred repos from GitHub.")
            repo_data = await fetch_starred_repos_with_docs(github_token)

        # 2. Process into doc texts + metadata
        pipeline_state["phase"] = "indexing"
        ids, metadata_list, doc_texts = await asyncio.to_thread(
            organize_repo_data, repo_data
        )

        # 3. Build in-memory Qdrant vector store (runs in thread to avoid blocking)
        def _build_vector_store():
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name="ask_my_bookmark",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            vs = QdrantVectorStore(
                client=client,
                collection_name="ask_my_bookmark",
                embedding=embeddings,
            )
            vs.add_texts(texts=doc_texts, ids=ids, metadatas=metadata_list)
            return vs

        vector_store = await asyncio.to_thread(_build_vector_store)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # 4. Compile LangGraph RAG pipeline
        rag_graph = build_rag_graph(retriever)

        pipeline_state.update(
            {
                "rag_graph": rag_graph,
                "status": "ready",
                "phase": None,
            }
        )
        print(
            f"Pipeline ready — {pipeline_state['total_repos']} repos indexed."
        )

    except Exception as exc:
        pipeline_state.update(
            {
                "status": "error",
                "phase": None,
                "error": str(exc),
            }
        )
        print(f"Pipeline build error: {exc}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SetupRequest(BaseModel):
    github_token: str


class QueryRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/setup")
async def setup(request: SetupRequest):
    if pipeline_state["status"] == "loading":
        return {
            "status": "loading",
            "message": "Pipeline is already being built.",
        }
    asyncio.create_task(_build_pipeline(request.github_token))
    return {"status": "loading", "message": "Pipeline build started."}


@app.get("/api/status")
async def status():
    return {
        "status": pipeline_state["status"],
        "phase": pipeline_state["phase"],
        "repo_count": pipeline_state["repo_count"],
        "total_repos": pipeline_state["total_repos"],
        "error": pipeline_state["error"],
    }


@app.post("/api/query")
async def query(request: QueryRequest):
    if pipeline_state["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline not ready. Current status: {pipeline_state['status']}",
        )
    rag_graph = pipeline_state["rag_graph"]
    result = await rag_graph.ainvoke({"question": request.question})
    return {"response": result["response"]}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.ask_my_bookmark:app", host="0.0.0.0", port=8000, reload=True)
