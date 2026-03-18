"""
Orchestrator graph for AskMyBookmark.

Ported from notebooks/orchestrator.ipynb.  All node functions are defined as
closures inside ``build_orchestrator_graph()`` so they capture the retriever
objects (search_df, vector_store) that are built by the pipeline at startup.
No globals — safe to run multiple graphs in the same process.
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import nltk
import pandas as pd
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, ConfigDict, Field
from searcharray import SearchArray
from typing import TypedDict

# Download NLTK data once at import time
for _pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVER_K = 15          # over-fetch pool for the reranker
CURATED_FILTER_THRESHOLD = 4.0
CURATED_LABEL_THRESHOLD  = 2.0
MAX_FEEDBACK_ROUNDS      = 3

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str

    # ── Query prep ─────────────────────────────────────────────────────────
    keywords:  List[str]
    expansions: List[Dict]
    bm25_terms: List[str]

    # ── Routing ────────────────────────────────────────────────────────────
    route:           str   # "lexical" | "ensemble"
    include_curated: bool

    # ── Retrieval ──────────────────────────────────────────────────────────
    bm25_results:    List[Document]
    vector_results:  List[Document]
    merged_results:  List[Document]

    # ── Output size ────────────────────────────────────────────────────────
    top_k: int

    # ── Final answer ───────────────────────────────────────────────────────
    answer: str

    # ── Feedback loop ──────────────────────────────────────────────────────
    feedback:           Dict[str, str]   # repo → "good"|"meh"|"bad"
    blocklist:          List[str]        # ☹️ repos never shown again
    good_repos:         List[Document]   # 😃 repos preserved across rounds
    feedback_iteration: int

# ---------------------------------------------------------------------------
# Pydantic models (structured LLM outputs)
# ---------------------------------------------------------------------------

class KeywordExpansion(BaseModel):
    keyword:  str       = Field(description="The original extracted keyword.")
    synonyms: List[str] = Field(
        description="Synonyms and closely related technical terms for this keyword "
                    "in the context of GitHub repos."
    )

class QueryPrepOutput(BaseModel):
    keywords:   List[str]            = Field(description=(
        "Signal-bearing keywords extracted from the query: repo names, technology "
        "names, programming languages, domain terms, and action verbs. Strip "
        "conversational filler, pronouns, stopwords, and words related to "
        "'repositories', 'repos', 'GitHub', 'favorites', or 'starred'."
    ))
    expansions: List[KeywordExpansion] = Field(description="Per-keyword synonym expansions.")
    bm25_terms: List[str]              = Field(description=(
        "Flat list of all synonyms collected from expansions, used to expand "
        "the BM25 query. Do not repeat the original keywords here."
    ))
    route: Literal["lexical", "ensemble"] = Field(description=(
        "Search strategy. Choose 'lexical' ONLY when the query is literally just "
        "bare keywords with no sentence structure, verbs, or question words. "
        "Choose 'ensemble' for everything else. When in doubt, choose 'ensemble'."
    ))
    include_curated: bool = Field(description=(
        "True ONLY when the user explicitly asks for lists, resources, collections, "
        "courses, or curated roundups. False when the user wants a specific tool, "
        "library, or implementation."
    ))
    reasoning: str = Field(description="One sentence explaining the route and include_curated decisions.")

class RerankedList(BaseModel):
    ranked_indices: List[int] = Field(description=(
        "1-based indices of the candidates ordered from most relevant to least "
        "relevant. Include every candidate index exactly once."
    ))

class _CuratedItem(BaseModel):
    index:          int  = Field(description="0-based index of the candidate in the input list")
    is_curated_list: bool = Field(description=(
        "True if the repo is a curated list/directory of external links; "
        "False if it is a real project delivering its own code, data, or content."
    ))
    reason: str = Field(description="Brief 5-10 word explanation for the classification")

class _CuratedClassifications(BaseModel):
    classifications: List[_CuratedItem] = Field(
        description="Classification for every candidate in the input list"
    )

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

QUERY_PREP_SYSTEM_PROMPT = """\
You are the query preparation agent for AskMyBookmark — a search assistant over a user's \
starred GitHub repositories.

Given a user query, perform four tasks in a single response:

## 1. Extract keywords
Extract only the signal-bearing terms: repo names, technology names, programming languages, \
domain terms, and action verbs. Strip conversational filler, pronouns, and stopwords. \
Ignore words related to "repositories", "repos", "GitHub", "favorites", or "starred".

## 2. Expand with synonyms
For each extracted keyword, generate relevant synonyms and closely related technical terms \
in the context of software and GitHub repositories.

## 3. Decide the search route
- Choose **lexical** ONLY when the user query is literally just bare keywords with no \
  sentence structure, verbs, or question words (e.g. "pytorch transformers cuda"). \
  This is rare.
- Choose **ensemble** for all natural language queries, conceptual queries, questions, \
  or anything with sentence structure. This is the default.
- When in doubt, always choose **ensemble**.

## 4. Decide include_curated
- Set to **True** ONLY when the user explicitly asks for lists, resources, collections, \
  courses, or curated roundups.
- Set to **False** (the default) when the user wants a specific tool, library, framework, \
  or implementation — not a collection of links.
"""

FEEDBACK_REFINEMENT_SYSTEM_PROMPT = """\
You are the query refinement agent for AskMyBookmark.

The user ran a search and gave emoji feedback on the results:

  😃 (good)  — These repos satisfy the user's need. Use their descriptions,
                topics, and names to generate new search terms that find MORE
                repos like these.

  😑 (meh)   — Tangentially relevant but not ideal. Try to surface better
                alternatives.

  ☹️ (bad)   — These actively frustrate the user. Analyse WHY they are wrong
                (wrong type? wrong language? wrong abstraction level? curated
                list when a real project was wanted?) and avoid those patterns.

Your job is to produce a refined QueryPrepOutput that:
1. Introduces new keywords and synonyms inspired by the 😃 repos.
2. Avoids vocabulary and topic patterns common to the ☹️ repos.
3. Updates route and include_curated if the feedback reveals a clear pattern.

Keep the original query intent — do not drift into unrelated topics.
"""

CURATED_CLASSIFIER_SYSTEM_PROMPT = """\
You are a classifier that determines whether a GitHub repository is a **curated list** \
or a **real project**.

## Key Principle
A **curated list** is a repository whose primary value is aggregating links or references \
to external content (other repos, papers, tools, tutorials, courses, books). It acts as an \
index or directory.

A **real project** delivers its own original code, data, models, or written content. Even if \
it uses the words "curated", "collection", or "list" in its description, what matters is \
whether the repo IS the thing or merely points to other things.

## Examples of CURATED LISTS
[0] Repo: wsvincent/awesome-django
    Description: A curated list of awesome things related to Django
    Topics: awesome, awesome-list, django
    → CURATED: aggregates links to Django tools, packages, and tutorials.

[1] Repo: eugeneyan/applied-ml
    Description: Papers & tech blogs by companies sharing their work on data science & machine learning in production.
    Topics: applied-machine-learning, data-science, natural-language-processing
    → CURATED: aggregates external blog posts and papers; the repo itself contains no original code.

[2] Repo: Developer-Y/cs-video-courses
    Description: List of Computer Science courses with video lectures.
    Topics: algorithms, computer-science, machine-learning
    → CURATED: a directory of links to external video courses.

[3] Repo: NirDiamant/RAG_Techniques
    Description: This repository showcases various advanced techniques for Retrieval-Augmented Generation (RAG) systems.
    Topics: langchain, llm, rag, tutorials
    → CURATED: despite sounding like a project, it is a collection of tutorial notebooks.

[4] Repo: ossu/computer-science
    Description: Path to a free self-taught education in Computer Science!
    Topics: computer-science, curriculum, education
    → CURATED: a structured roadmap of links to external free courses.

[5] Repo: ashishps1/learn-ai-engineering
    Description: Learn AI and LLMs from scratch using free resources
    Topics: ai, large-language-models, llm, machine-learning
    → CURATED: primary value is the aggregated collection of free external resources.

[6] Repo: EbookFoundation/free-programming-books
    Description: Freely available programming books
    Topics: books, education, list
    → CURATED: a pure link collection; the "content" is pointers to books available elsewhere.

[7] Repo: codecrafters-io/build-your-own-x
    Description: Master programming by recreating your favorite technologies from scratch.
    Topics: programming, tutorial
    → CURATED: aggregates links to external "build your own X" tutorials.

## Examples of REAL PROJECTS

[8] Repo: explosion/curated-transformers
    Description: A PyTorch library of curated Transformer models and their composable components
    Topics: bert, llm, nlp, pytorch, transformer
    → REAL PROJECT: despite "curated" in the name, this IS a usable PyTorch library.

[9] Repo: google-deepmind/mujoco_menagerie
    Description: A collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind.
    Topics: mujoco, robotics
    → REAL PROJECT: the model files ARE the deliverable; "curated" describes quality.

[10] Repo: h5bp/Front-end-Developer-Interview-Questions
     Description: A list of helpful front-end related questions you can use to interview potential candidates.
     Topics: css-questions, front-end, html-questions, interview-questions
     → REAL PROJECT: the questions themselves are the original written content.

[11] Repo: faridrashidi/kaggle-solutions
     Description: Collection of Kaggle Solutions and Ideas
     Topics: kaggle, machine-learning, solutions
     → REAL PROJECT: contains actual solution code notebooks.

[12] Repo: openai/spinningup
     Description: An educational resource to help anyone learn deep reinforcement learning.
     Topics: deep-reinforcement-learning, machine-learning, reinforcement-learning
     → REAL PROJECT: contains actual RL algorithm implementations in code.

[13] Repo: huggingface/transformers
     Description: The model-definition framework for state-of-the-art machine learning models in text, vision, audio.
     Topics: deep-learning, llm, machine-learning, nlp, pytorch
     → REAL PROJECT: a full ML framework delivering original software.

[14] Repo: weaviate/weaviate
     Description: Weaviate is an open-source vector database that stores both objects and vectors.
     Topics: vector-database, vector-search, semantic-search
     → REAL PROJECT: a deployable database system with its own code.

[15] Repo: argilla-io/argilla
     Description: Argilla is a collaboration tool for AI engineers and domain experts to build high-quality datasets
     Topics: annotation-tool, llm, nlp, text-labeling
     → REAL PROJECT: a specific tool with real application code — not a link directory.

## Task
For each candidate below, classify it as a curated list or a real project.
Output JSON with a "classifications" array covering EVERY candidate index provided.
"""

RERANKER_SYSTEM_PROMPT = """\
You are a search result reranker for AskMyBookmark, an assistant that searches a user's
starred GitHub repositories.

You will be given a user query and a numbered list of repository candidates.
Each candidate includes a "Curated list" field that tells you whether the repository is a
curated collection of links/resources (Yes) or an actual project/library/tool (Most likely not).

Your task is to reorder the candidates from most relevant to least relevant for the query.

Guidelines:
- Consider the repo name, description, topics, programming language, and curated-list status.
- A candidate is highly relevant if it directly addresses the query's intent.
- When candidates are closely matched, prefer repos that are more specific to the query.
- Curated list handling:
    * If the query is asking for a specific tool, library, or implementation (the common case),
      rank curated lists BELOW specific projects of equal relevance.
    * If the query is explicitly asking for resources, collections, courses, or roundups,
      rank relevant curated lists higher.
- Prior user feedback (when present):
    * 😃 Good — the user liked this result; rank it near the top.
    * 😑 Meh  — the user was neutral; rank in the middle unless clearly better than others.
    * ☹️ Bad  — the user disliked this; rank it at the bottom.
- Return ALL candidate indices in your ranked list — do not omit any.
"""

RAG_SYSTEM_PROMPT = """\
You are AskMyBookmark, a personal research assistant with access to the user's GitHub starred repositories.

Your job is to help the user discover, recall, and explore repositories they have bookmarked on GitHub. \
You answer questions by reasoning over the retrieved repository context provided to you — not from your \
general knowledge of what exists on GitHub.

**Ground rules:**
- Only surface repositories that appear in the retrieved context below. Do not invent or suggest \
  repositories that are not present in the context.
- If no retrieved repositories are relevant to the query, say so honestly and suggest the user try \
  rephrasing or broadening their search.
- You may use your general knowledge to explain a topic or technology, but all repository \
  recommendations must come exclusively from the retrieved context.

**Format your response as a strict numbered list from 1 to {n_results}.**
- Every number from 1 to {n_results} must appear exactly once. Do not skip any number.
- Each numbered entry must correspond to exactly one repository from the context, in the order \
  they are listed (already ranked best-first).
- Do not add a closing paragraph summarising which repos were "most relevant" or noting that \
  others were omitted — every entry must appear in the numbered list.

**For each numbered entry include:**
- The repository's full name as a markdown link: [owner/repo](URL)
- A brief description of what it does, in your own words.
- 1–2 sentences on *why* it is relevant to the user's query.
- Optionally: language, star count, or topics if they help the user evaluate the match.
"""

RAG_HUMAN_PROMPT_TEMPLATE = """\
User query: {query}

Retrieved repositories ({n_results} total, already ranked best-first):
Format your answer as a numbered list from 1 to {n_results}. Every number must appear.
{context}
"""

# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_text(text: str | None) -> List[str]:
    """Tokenizer for SearchArray BM25 — lowercase, remove stopwords, lemmatize."""
    if text is None:
        return []
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    _stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens if w not in _stop and len(w) > 2]

# ---------------------------------------------------------------------------
# MultiMatchBM25Retriever
# ---------------------------------------------------------------------------

def _multi_match_search(
    query: str,
    df: pd.DataFrame,
    columns: List[str],
    boosts: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """BM25 multi-field search with per-field boost weights and dismax scoring."""
    boosts = boosts or {}
    boost_values = {col: boosts.get(col, 1.0) for col in columns}
    tokenized_queries = {col: df[col].array.tokenizer(query) for col in columns}
    field_scores = {
        col: np.asarray([df[col].array.score(term) for term in tokenized_queries[col]])
             * boost_values[col]
        for col in columns
    }
    num_terms = max((len(s) for s in field_scores.values()), default=0)
    if num_terms == 0:
        result = df.copy()
        result["score"] = 0.0
        return result
    best_term_scores = []
    for term_idx in range(num_terms):
        term_scores = [
            field_scores[col][term_idx]
            for col in columns
            if term_idx < len(field_scores[col])
        ]
        best_term_scores.append(np.max(term_scores, axis=0))
    result = df.copy()
    result["score"] = np.sum(best_term_scores, axis=0)
    return result.sort_values("score", ascending=False)


class MultiMatchBM25Retriever(BaseRetriever):
    """LangChain-compatible retriever wrapping SearchArray multi-field BM25."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_df: pd.DataFrame
    columns:   List[str]
    boosts:    Dict[str, float]
    k:         int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        results = _multi_match_search(
            query=query, df=self.search_df, columns=self.columns, boosts=self.boosts
        )
        top_k = results[results["score"] > 0].head(self.k)
        return [
            Document(
                page_content=row["content"],
                metadata={
                    "id":                row["id"],
                    "repo":              row["repo"],
                    "description":       row["description"],
                    "topics":            row.get("topics_list", row["topics"].split()),
                    "language":          row.get("language"),
                    "stars":             row.get("stars"),
                    "url":               row.get("url"),
                    "curated_list_bm25": float(row.get("curated_list_bm25", 0.0)),
                    "score":             float(row["score"]),
                },
            )
            for _, row in top_k.iterrows()
        ]

# ---------------------------------------------------------------------------
# Regex fast-path patterns for curated-list classification
# ---------------------------------------------------------------------------

_CURATED_POSITIVE_DESC_RE = re.compile(
    r"(?:"
    r"curated list of"
    r"|\blist of\b.{0,100}\b(?:tools|resources|tutorials|papers|courses|links|examples|"
    r"projects|repos|repositories|books|videos|algorithms|frameworks|libraries|datasets)\b"
    r"|\bcollection of\b.{0,100}\b(?:resources|tutorials|papers|tools|links|examples|"
    r"courses|books|videos)\b"
    r"|\bfreely available\b.{0,40}\bbooks\b"
    r"|\b\d{2,4}\+?\s+\w[\w\s,/]+projects with code\b"
    r"|\bindex of\b.{0,80}\b(?:algorithms|papers|resources|tools|courses)\b"
    r")",
    re.IGNORECASE,
)
_CURATED_POSITIVE_TOPICS  = frozenset({"awesome-list", "curated-list", "curated-lists"})
_CURATED_AWESOME_REPO_RE  = re.compile(r"^[^/]+/awesome[-_]", re.IGNORECASE)
_NOT_CURATED_DESC_RE      = re.compile(
    r"(?:"
    r"\bis (?:an? )?(?:open.?source |fast |lightweight |simple )?(?:library|framework|engine|database|platform|sdk|cli|toolkit)\b"
    r"|official (?:pytorch|tensorflow|jax|keras) implementation\b"
    r"|\bpython (?:library|package|wrapper|client)\b"
    r"|workflow engine\b"
    r")",
    re.IGNORECASE,
)
_WEAK_POSITIVE_DESC_RE = re.compile(
    r"\b(?:list|collection|resource|tutorial|paper|course|curated|survey|index|roundup)\b",
    re.IGNORECASE,
)


def _quick_curated_check(m: dict) -> Optional[bool]:
    """Regex fast-path: returns True (curated), False (real project), or None (ambiguous)."""
    repo        = m.get("repo", "")
    description = (m.get("description") or "").strip()
    topics: List[str] = m.get("topics") or []

    if _CURATED_AWESOME_REPO_RE.match(repo):
        return True
    if _CURATED_POSITIVE_TOPICS.intersection(topics):
        return True
    if description and _CURATED_POSITIVE_DESC_RE.search(description):
        return True
    if description and _NOT_CURATED_DESC_RE.search(description):
        if not _WEAK_POSITIVE_DESC_RE.search(description):
            return False
    return None


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _curated_label(m: dict) -> str:
    if "is_curated_llm" in m:
        return "Yes" if m["is_curated_llm"] else "Most likely not"
    score = m.get("curated_list_bm25", 0.0)
    if score >= CURATED_FILTER_THRESHOLD:
        return "Yes"
    if score >= CURATED_LABEL_THRESHOLD:
        return "Likely"
    return "Most likely not"


def _format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        m        = doc.metadata
        repo     = m.get("repo", "unknown")
        url      = m.get("url") or f"https://github.com/{repo}"
        language = m.get("language") or "unknown"
        stars    = m.get("stars")
        stars_str = f"{stars:,}" if isinstance(stars, (int, float)) and stars else "unknown"
        parts.append(
            f"Repo: {repo}\n"
            f"URL: {url}\n"
            f"Description: {m.get('description', '')}\n"
            f"Topics: {', '.join(m.get('topics', []))}\n"
            f"Language: {language}\n"
            f"Stars: {stars_str}\n"
            f"Curated list: {_curated_label(m)}\n"
            "---"
        )
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_orchestrator_graph(
    search_df:    pd.DataFrame,
    vector_store: Any,
    checkpointer: Any = None,
):
    """Compile the full orchestrator LangGraph.

    Parameters
    ----------
    search_df:
        DataFrame built by the pipeline (must already have repo_idx, description_idx,
        topics_idx, content_idx SearchArray columns and curated_list_bm25 column).
    vector_store:
        QdrantVectorStore (already indexed).
    checkpointer:
        Optional LangGraph checkpointer (e.g. MemorySaver) for interrupt/resume.
    """

    # ── Build retrievers ────────────────────────────────────────────────────
    bm25_retriever = MultiMatchBM25Retriever(
        search_df=search_df,
        columns=["repo_idx", "topics_idx", "description_idx", "content_idx"],
        boosts={"repo_idx": 3.0, "topics_idx": 2.0, "description_idx": 1.5, "content_idx": 1.0},
        k=RETRIEVER_K,
    )

    legacy_docs = [
        Document(
            page_content=row["content"],
            metadata={
                "id":                row["id"],
                "repo":              row["repo"],
                "description":       row["description"],
                "topics":            row.get("topics_list", []),
                "language":          row.get("language"),
                "stars":             row.get("stars"),
                "url":               row.get("url"),
                "curated_list_bm25": float(row.get("curated_list_bm25", 0.0)),
            },
        )
        for _, row in search_df.iterrows()
    ]
    legacy_retriever = BM25Retriever.from_documents(legacy_docs, k=RETRIEVER_K)

    vector_retriever   = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    # ── LLM instances ──────────────────────────────────────────────────────
    _query_prep_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        QueryPrepOutput
    )
    _reranker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        RerankedList
    )
    _curator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        _CuratedClassifications
    )
    _rag_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(RAG_HUMAN_PROMPT_TEMPLATE),
    ])
    _rag_chain = _rag_prompt | ChatOpenAI(model="gpt-4o-mini")

    # ── Node: query_prep ────────────────────────────────────────────────────
    def query_prep(state: OrchestratorState) -> dict:
        result: QueryPrepOutput = _query_prep_llm.invoke([
            SystemMessage(content=QUERY_PREP_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ])
        bm25_terms = result.bm25_terms or [
            syn for exp in result.expansions for syn in exp.synonyms
        ]
        return {
            "keywords":        result.keywords,
            "expansions":      [exp.model_dump() for exp in result.expansions],
            "bm25_terms":      bm25_terms,
            "route":           result.route,
            "include_curated": result.include_curated,
        }

    # ── Node: refine_query ──────────────────────────────────────────────────
    def refine_query(state: OrchestratorState) -> dict:
        feedback = state.get("feedback") or {}
        merged   = state.get("merged_results") or []
        good = [d for d in merged if feedback.get(d.metadata.get("repo")) == "good"]
        bad  = [d for d in merged if feedback.get(d.metadata.get("repo")) == "bad"]
        meh  = [d for d in merged if feedback.get(d.metadata.get("repo")) == "meh"]

        def _repo_line(doc: Document) -> str:
            m = doc.metadata
            return (
                f"  - {m.get('repo', 'unknown')}: {m.get('description', 'N/A')}\n"
                f"    Topics: {', '.join(m.get('topics', []))}"
            )

        parts = [f"Original query: {state['query']}\n"]
        if good:
            parts.append("😃 REPOS THE USER LIKED (find more like these):")
            parts.extend(_repo_line(d) for d in good)
        if bad:
            parts.append("\n☹️ REPOS THE USER DISLIKED (avoid these patterns):")
            parts.extend(_repo_line(d) for d in bad)
        if meh:
            parts.append("\n😑 REPOS THE USER WAS MEH ABOUT (try to find better):")
            parts.extend(_repo_line(d) for d in meh)

        result: QueryPrepOutput = _query_prep_llm.invoke([
            SystemMessage(content=FEEDBACK_REFINEMENT_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(parts)),
        ])
        bm25_terms = result.bm25_terms or [
            syn for exp in result.expansions for syn in exp.synonyms
        ]
        return {
            "keywords":        result.keywords,
            "expansions":      [exp.model_dump() for exp in result.expansions],
            "bm25_terms":      bm25_terms,
            "route":           result.route,
            "include_curated": result.include_curated,
        }

    # ── Routing helper ──────────────────────────────────────────────────────
    def route_after_query_prep(state: OrchestratorState) -> str:
        return {"lexical": "lexical_search", "ensemble": "ensemble_search"}[state["route"]]

    # ── Nodes: search ───────────────────────────────────────────────────────
    def _expanded_query(state: OrchestratorState) -> str:
        return " ".join(state["keywords"] + state["bm25_terms"])

    def lexical_search(state: OrchestratorState) -> dict:
        q = _expanded_query(state)
        multi   = bm25_retriever.invoke(q)
        legacy  = legacy_retriever.invoke(q)
        seen    = {doc.metadata.get("repo") for doc in multi}
        extra   = [d for d in legacy if d.metadata.get("repo") not in seen]
        return {"bm25_results": multi + extra, "vector_results": []}

    def ensemble_search(state: OrchestratorState) -> dict:
        results = ensemble_retriever.invoke(_expanded_query(state))
        return {"bm25_results": results, "vector_results": []}

    # ── Node: merge_results ─────────────────────────────────────────────────
    def merge_results(state: OrchestratorState) -> dict:
        good_repos: List[Document] = state.get("good_repos") or []
        seen: set = {doc.metadata.get("repo") for doc in good_repos}
        merged: List[Document] = list(good_repos)
        for doc in (state["bm25_results"] + state["vector_results"]):
            repo = doc.metadata.get("repo")
            if repo and repo not in seen:
                seen.add(repo)
                merged.append(doc)
        return {"merged_results": merged}

    # ── Node: classify_curated ──────────────────────────────────────────────
    def classify_curated(state: OrchestratorState) -> dict:
        if state.get("include_curated", False):
            return {}
        candidates = state["merged_results"]
        if not candidates:
            return {}

        classified = [
            Document(page_content=d.page_content, metadata=dict(d.metadata))
            for d in candidates
        ]

        ambiguous_indices = []
        for i, doc in enumerate(classified):
            result = _quick_curated_check(doc.metadata)
            if result is True:
                doc.metadata["is_curated_llm"]    = True
                doc.metadata["is_curated_reason"] = "regex: obvious curated signal"
            elif result is False:
                doc.metadata["is_curated_llm"]    = False
                doc.metadata["is_curated_reason"] = "regex: obvious real project"
            else:
                ambiguous_indices.append(i)

        if ambiguous_indices:
            candidate_lines = []
            for i in ambiguous_indices:
                m = classified[i].metadata
                candidate_lines.append(
                    f"[{i}] Repo: {m.get('repo', 'unknown')}\n"
                    f"     Description: {m.get('description', '') or 'N/A'}\n"
                    f"     Topics: {', '.join(m.get('topics', [])) or 'none'}"
                )
            try:
                llm_result: _CuratedClassifications = _curator_llm.invoke([
                    SystemMessage(content=CURATED_CLASSIFIER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Candidates to classify:\n\n{chr(10).join(candidate_lines)}"),
                ])
                for item in llm_result.classifications:
                    if item.index in ambiguous_indices:
                        classified[item.index].metadata["is_curated_llm"]    = item.is_curated_list
                        classified[item.index].metadata["is_curated_reason"] = item.reason
            except Exception:
                pass  # Fall back to BM25-based label

        return {"merged_results": classified}

    # ── Node: filter_results ────────────────────────────────────────────────
    def filter_results(state: OrchestratorState) -> dict:
        if state.get("include_curated", False):
            return {"merged_results": state["merged_results"]}

        candidates  = state["merged_results"]
        top_k       = state.get("top_k", 10)
        safety_floor = min(top_k, 3)
        blocklist   = set(state.get("blocklist") or [])

        filtered = [
            doc for doc in candidates
            if doc.metadata.get("repo") not in blocklist
            and not doc.metadata.get("is_curated_llm", False)
            and doc.metadata.get("curated_list_bm25", 0.0) < CURATED_FILTER_THRESHOLD
        ]

        if len(filtered) < safety_floor:
            relaxed = [d for d in candidates if d.metadata.get("repo") not in blocklist]
            if len(relaxed) >= safety_floor:
                return {"merged_results": relaxed}
            return {"merged_results": candidates}

        return {"merged_results": filtered}

    # ── Node: rerank_results ────────────────────────────────────────────────
    def rerank_results(state: OrchestratorState) -> dict:
        candidates = state["merged_results"]
        top_k      = state.get("top_k", 10)
        if not candidates:
            return {"merged_results": []}

        feedback = state.get("feedback") or {}
        formatted = []
        for i, doc in enumerate(candidates, 1):
            m     = doc.metadata
            repo  = m.get("repo", "unknown")
            prior = feedback.get(repo)
            rating_line = ""
            if prior == "good":
                rating_line = "\n   Prior feedback: 😃 Good"
            elif prior == "meh":
                rating_line = "\n   Prior feedback: 😑 Meh"
            elif prior == "bad":
                rating_line = "\n   Prior feedback: ☹️ Bad (rank lower)"
            formatted.append(
                f"{i}. Repo: {repo}\n"
                f"   Description: {m.get('description', '') or 'N/A'}\n"
                f"   Topics: {', '.join(m.get('topics', [])) or 'none'}\n"
                f"   Language: {m.get('language', '') or 'N/A'}\n"
                f"   Curated list: {_curated_label(m)}"
                + rating_line
            )

        user_msg = f"Query: {state['query']}\n\nCandidates:\n\n" + "\n\n".join(formatted)
        try:
            result: RerankedList = _reranker_llm.invoke([
                SystemMessage(content=RERANKER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            seen:     set           = set()
            reranked: List[Document] = []
            for idx in result.ranked_indices:
                if 1 <= idx <= len(candidates) and idx not in seen:
                    seen.add(idx)
                    reranked.append(candidates[idx - 1])
            for i, doc in enumerate(candidates, 1):
                if i not in seen:
                    reranked.append(doc)
            return {"merged_results": reranked[:top_k]}
        except Exception:
            return {"merged_results": candidates[:top_k]}

    # ── Node: generate_answer ───────────────────────────────────────────────
    def generate_answer(state: OrchestratorState) -> dict:
        docs     = state["merged_results"]
        context  = _format_context(docs)
        response = _rag_chain.invoke({
            "query":     state["query"],
            "context":   context,
            "n_results": len(docs),
        })
        return {"answer": response.content}

    # ── Node: human_feedback ────────────────────────────────────────────────
    def human_feedback(state: OrchestratorState) -> dict:
        """Interrupt and wait for emoji feedback via Command(resume=ratings)."""
        ratings: dict = interrupt("Waiting for user feedback")

        docs_by_repo = {doc.metadata.get("repo"): doc for doc in state["merged_results"]}

        prev_good: List[Document] = state.get("good_repos") or []
        prev_good_repos: set = {doc.metadata.get("repo") for doc in prev_good}
        new_good = [
            docs_by_repo[repo]
            for repo, rating in ratings.items()
            if rating == "good" and repo in docs_by_repo and repo not in prev_good_repos
        ]
        good_repos = prev_good + new_good

        prev_blocklist: List[str] = state.get("blocklist") or []
        new_bad = [repo for repo, rating in ratings.items() if rating == "bad"]
        blocklist = list(set(prev_blocklist + new_bad))

        return {
            "feedback":           ratings,
            "good_repos":         good_repos,
            "blocklist":          blocklist,
            "feedback_iteration": (state.get("feedback_iteration") or 0) + 1,
        }

    # ── Conditional edge: after feedback ───────────────────────────────────
    def should_continue_after_feedback(state: OrchestratorState) -> str:
        feedback  = state.get("feedback") or {}
        iteration = state.get("feedback_iteration") or 0

        if feedback.get("__stop"):
            return END
        if iteration > MAX_FEEDBACK_ROUNDS:
            return END
        has_unsatisfied = any(
            v in ("bad", "meh")
            for k, v in feedback.items()
            if not k.startswith("__")
        )
        if not has_unsatisfied:
            return END
        return "refine_query"

    # ── Assemble graph ──────────────────────────────────────────────────────
    graph = StateGraph(OrchestratorState)

    graph.add_node("query_prep",       query_prep)
    graph.add_node("refine_query",     refine_query)
    graph.add_node("lexical_search",   lexical_search)
    graph.add_node("ensemble_search",  ensemble_search)
    graph.add_node("merge_results",    merge_results)
    graph.add_node("classify_curated", classify_curated)
    graph.add_node("filter_results",   filter_results)
    graph.add_node("rerank_results",   rerank_results)
    graph.add_node("generate_answer",  generate_answer)
    graph.add_node("human_feedback",   human_feedback)

    graph.add_edge(START, "query_prep")

    _search_routing = {"lexical_search": "lexical_search", "ensemble_search": "ensemble_search"}
    graph.add_conditional_edges("query_prep",    route_after_query_prep, _search_routing)
    graph.add_conditional_edges("refine_query",  route_after_query_prep, _search_routing)

    graph.add_edge("lexical_search",   "merge_results")
    graph.add_edge("ensemble_search",  "merge_results")
    graph.add_edge("merge_results",    "classify_curated")
    graph.add_edge("classify_curated", "filter_results")
    graph.add_edge("filter_results",   "rerank_results")
    graph.add_edge("rerank_results",   "generate_answer")
    graph.add_edge("generate_answer",  "human_feedback")

    graph.add_conditional_edges(
        "human_feedback",
        should_continue_after_feedback,
        {"refine_query": "refine_query", END: END},
    )

    return graph.compile(checkpointer=checkpointer)
