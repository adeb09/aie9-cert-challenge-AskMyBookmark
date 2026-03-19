import Head from "next/head";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PipelineStatus = "idle" | "loading" | "ready" | "error";
type LoadingPhase   = "fetching" | "indexing" | null;
type EmojiRating    = "bad" | "meh" | "good";

interface ProgressUpdate {
  label:      string;
  step:       number;
  totalSteps: number;
}

interface StatusResponse {
  status:          PipelineStatus;
  phase:           LoadingPhase;
  github_username: string | null;
  fetch_step:      string | null;
  repo_count:      number;
  total_repos:     number;
  index_step:      string | null;
  index_count:     number;
  index_total:     number;
  error:           string | null;
}

interface RepoResult {
  repo:        string;
  url:         string;
  description: string;
  language:    string;
  stars:       number | null;
  topics:      string[];
}

interface SessionResponse {
  session_id: string;
  answer:     string;
  results:    RepoResult[];
  iteration:  number;
  done:       boolean;
}

interface SearchRound {
  question:  string;
  answer:    string;
  results:   RepoResult[];
  iteration: number;
  sessionId: string | null;  // null = session done
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EMOJI: Record<EmojiRating, string> = {
  bad:  "☹️",
  meh:  "😑",
  good: "😃",
};

const RATINGS: EmojiRating[] = ["bad", "meh", "good"];

/**
 * Split the LLM's numbered markdown answer into per-item strings.
 * Returns { intro, items } where intro is any text before "1. …"
 * and items is one markdown string per numbered result.
 */
function parseAnswerItems(answer: string): { intro: string; items: string[] } {
  // Find the position of the first numbered list item ("1. ")
  const firstMatch = answer.match(/^1\.\s/m);
  if (!firstMatch || firstMatch.index === undefined) {
    return { intro: answer, items: [] };
  }
  const intro    = answer.slice(0, firstMatch.index).trim();
  const listPart = answer.slice(firstMatch.index);
  // Split on lines that begin a new numbered item (e.g. "\n2. ")
  const items = listPart
    .split(/\n(?=\d+\.\s)/)
    .map((s) => s.trim())
    .filter((s) => /^\d+\.\s/.test(s));
  return { intro, items };
}

/**
 * Read a text/event-stream response line-by-line and call onEvent for each
 * parsed "data: {...}" line.  Works with fetch() POST responses (unlike EventSource).
 */
async function readSSEStream(
  response: Response,
  onEvent: (data: Record<string, unknown>) => void,
): Promise<void> {
  const reader  = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer    = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ") && line.length > 6) {
          try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip malformed */ }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

const FUNNY_MESSAGES = [
  "Dusting off your starred repos…",
  "Judging repos by their READMEs…",
  "Separating the awesome-lists from the actually awesome…",
  "Counting stars (the GitHub kind)…",
  "Reading someone else's README so you don't have to…",
  "Debating whether to include the 'awesome-' repos…",
  "Cross-referencing your impeccable taste in repos…",
  "Trying to remember why you starred that one repo…",
  "Measuring semantic distance in 1536 dimensions…",
  "Consulting the oracle of BM25…",
  "Asking the embeddings nicely…",
  "Your future stack is being computed…",
  "Summoning the power of keyword expansion…",
  "Calculating cosine similarity with enthusiasm…",
  "Ranking things that were already ranked…",
  "Blurmflurping your query into the void…",
  "Splutterglooping through the vector space…",
  "Flooperdoodling the BM25 index…",
  "Flibbertigibbeting the synonyms…",
  "Discombobulating the curated lists…",
  "Wizarding up some keyword expansions…",
  "Booping the embeddings into shape…",
  "Meandering through your 1,000+ stars…",
  "Sussing out the most relevant repos…",
  "Jiving with the reranker…",
];

const TOTAL_QUERY_STEPS = 6;
const MAX_ROUNDS = 3;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Home() {
  // ── Setup state ────────────────────────────────────────────────────────────
  const [token, setToken]               = useState("");
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>("idle");
  const [phase, setPhase]               = useState<LoadingPhase>(null);
  const [repoCount, setRepoCount]       = useState(0);
  const [totalRepos, setTotalRepos]     = useState(0);
  const [setupError, setSetupError]     = useState<string | null>(null);
  const [githubUsername, setGithubUsername] = useState<string | null>(null);
  const [fetchStep, setFetchStep]           = useState<string | null>(null);
  const [indexStep, setIndexStep]       = useState<string | null>(null);
  const [indexCount, setIndexCount]     = useState(0);
  const [indexTotal, setIndexTotal]     = useState(0);
  // null = no prompt shown yet; true/false = cache found/not found, awaiting user choice
  const [cachePrompt, setCachePrompt]   = useState<boolean | null>(null);

  // ── Query state ────────────────────────────────────────────────────────────
  const [question, setQuestion]         = useState("");
  const [isSearching, setIsSearching]   = useState(false);
  const [queryError, setQueryError]     = useState<string | null>(null);
  const [queryProgress, setQueryProgress] = useState<ProgressUpdate | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState("");

  // ── Session / feedback state ───────────────────────────────────────────────
  const [currentSession, setCurrentSession]   = useState<SessionResponse | null>(null);
  const [ratings, setRatings]                 = useState<Record<string, EmojiRating>>({});
  const [isRefining, setIsRefining]           = useState(false);
  const [refineError, setRefineError]         = useState<string | null>(null);

  // History of completed rounds (shown above current)
  const [history, setHistory]                 = useState<SearchRound[]>([]);

  const resultsRef = useRef<HTMLDivElement>(null);

  // ── Rotating funny message during search ───────────────────────────────────
  const [funnyMsgIdx, setFunnyMsgIdx] = useState(0);

  useEffect(() => {
    if (!isSearching && !isRefining) return;
    // Pick a random starting message each time a search begins
    setFunnyMsgIdx(Math.floor(Math.random() * FUNNY_MESSAGES.length));
    const id = setInterval(() => {
      setFunnyMsgIdx((i) => (i + 1) % FUNNY_MESSAGES.length);
    }, 2500);
    return () => clearInterval(id);
  }, [isSearching, isRefining]);

  // ── Polling ────────────────────────────────────────────────────────────────
  async function fetchStatus() {
    try {
      const res  = await fetch(`${API_BASE}/api/status`);
      if (!res.ok) return;
      const data: StatusResponse = await res.json();
      setPipelineStatus(data.status);
      setPhase(data.phase);
      setRepoCount(data.repo_count);
      setTotalRepos(data.total_repos);
      setGithubUsername(data.github_username);
      setFetchStep(data.fetch_step);
      setIndexStep(data.index_step);
      setIndexCount(data.index_count);
      setIndexTotal(data.index_total);
      if (data.status === "error") setSetupError(data.error ?? "An unknown error occurred.");
    } catch { /* backend not yet reachable */ }
  }

  useEffect(() => { fetchStatus(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (pipelineStatus !== "loading") return;
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [pipelineStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Scroll to new results ──────────────────────────────────────────────────
  useEffect(() => {
    if (currentSession) {
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    }
  }, [currentSession?.session_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Handlers ───────────────────────────────────────────────────────────────
  const handleSetup = async () => {
    if (!token.trim()) return;
    setSetupError(null);
    setCachePrompt(null);
    try {
      // Check whether a query cache already exists for this token's GitHub account.
      const res  = await fetch(`${API_BASE}/api/setup/check`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_token: token.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Setup check failed.");
      if (data.has_query_cache) {
        // Show the cache prompt and wait for the user's choice.
        setCachePrompt(true);
      } else {
        // No cache to offer — proceed immediately.
        await _doSetup(true);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSetupError(msg);
    }
  };

  const _doSetup = async (useCache: boolean) => {
    setCachePrompt(null);
    setPipelineStatus("loading");
    setPhase("fetching");
    setRepoCount(0);
    setTotalRepos(0);
    try {
      const res  = await fetch(`${API_BASE}/api/setup`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_token: token.trim(), use_cache: useCache }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Setup failed.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSetupError(msg);
      setPipelineStatus("error");
    }
  };

  const handleReset = () => {
    setPipelineStatus("idle");
    setPhase(null); setRepoCount(0); setTotalRepos(0);
    setGithubUsername(null); setFetchStep(null);
    setIndexStep(null); setIndexCount(0); setIndexTotal(0);
    setSetupError(null); setToken(""); setCachePrompt(null);
    setCurrentSession(null); setHistory([]); setRatings({});
    setQuestion(""); setStreamingAnswer("");
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isSearching) return;

    // Archive any open session first
    if (currentSession) {
      setHistory((h) => [
        { question: currentSession.results.length > 0 ? question : "Previous search",
          answer: currentSession.answer, results: currentSession.results,
          iteration: currentSession.iteration, sessionId: null },
        ...h,
      ]);
    }

    setIsSearching(true);
    setQueryError(null);
    setCurrentSession(null);
    setRatings({});
    setQueryProgress(null);
    setStreamingAnswer("");

    try {
      const res = await fetch(`${API_BASE}/api/session/start`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), top_k: 10 }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Search failed.");
      }
      await readSSEStream(res, (data) => {
        if (data.type === "progress") {
          setQueryProgress({
            label:      data.label      as string,
            step:       data.step       as number,
            totalSteps: data.total_steps as number,
          });
        } else if (data.type === "token") {
          setStreamingAnswer((prev) => prev + (data.text as string));
        } else if (data.type === "result") {
          setStreamingAnswer("");
          const session = data as unknown as SessionResponse;
          setCurrentSession(session);
          const defaultRatings: Record<string, EmojiRating> = {};
          session.results.forEach((r) => { defaultRatings[r.repo] = "meh"; });
          setRatings(defaultRatings);
        } else if (data.type === "error") {
          setQueryError((data.error as string) ?? "Search failed.");
        }
      });
    } catch (err: unknown) {
      setQueryError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSearching(false);
      setQueryProgress(null);
    }
  };

  const handleRatingChange = (repo: string, rating: EmojiRating) => {
    setRatings((prev) => ({ ...prev, [repo]: rating }));
  };

  const handleSubmitFeedback = async (done: boolean) => {
    if (!currentSession) return;
    setIsRefining(true);
    setRefineError(null);
    setQueryProgress(null);
    setStreamingAnswer("");

    // Archive the current round immediately so the user can see it scrolled up
    setHistory((h) => [
      { question: "", answer: currentSession.answer, results: currentSession.results,
        iteration: currentSession.iteration, sessionId: null },
      ...h,
    ]);

    try {
      const res = await fetch(`${API_BASE}/api/session/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSession.session_id, ratings, done }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Feedback failed.");
      }
      await readSSEStream(res, (data) => {
        if (data.type === "progress") {
          setQueryProgress({
            label:      data.label      as string,
            step:       data.step       as number,
            totalSteps: data.total_steps as number,
          });
        } else if (data.type === "token") {
          setStreamingAnswer((prev) => prev + (data.text as string));
        } else if (data.type === "result") {
          setStreamingAnswer("");
          const session = data as unknown as SessionResponse;
          if (session.done) {
            setCurrentSession({ ...session, done: true });
            setRatings({});
          } else {
            setCurrentSession(session);
            const newRatings: Record<string, EmojiRating> = {};
            session.results.forEach((r) => { newRatings[r.repo] = "meh"; });
            setRatings(newRatings);
          }
        } else if (data.type === "error") {
          setRefineError((data.error as string) ?? "Refinement failed.");
        }
      });
    } catch (err: unknown) {
      setRefineError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsRefining(false);
      setQueryProgress(null);
    }
  };

  // ── Derived state ──────────────────────────────────────────────────────────
  const progressPct = totalRepos > 0 ? Math.round((repoCount / totalRepos) * 100) : 0;

  const statusLabel: Record<PipelineStatus, string> = {
    idle:    "Not loaded",
    loading: "Loading…",
    ready:   githubUsername
               ? `@${githubUsername} · ${totalRepos.toLocaleString()} repos indexed`
               : `${totalRepos.toLocaleString()} repos indexed`,
    error:   "Error",
  };

  const badgeClass: Record<PipelineStatus, string> = {
    idle:    "badge badge-idle",
    loading: "badge badge-loading",
    ready:   "badge badge-ready",
    error:   "badge badge-error",
  };

  const canRefine = currentSession
    && !currentSession.done
    && currentSession.iteration < MAX_ROUNDS;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>AskMyBookmark</title>
        <meta name="description" content="Query your GitHub starred repositories with AI" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="page">
        {/* Header */}
        <header className="header">
          <h1>Ask<span>My</span>Bookmark</h1>
          <p>Conversational search over your starred GitHub repositories</p>
        </header>

        {/* ── Setup card ── */}
        {pipelineStatus !== "ready" && (
          <section className="card">
            <h2>Connect your GitHub account</h2>
            {pipelineStatus === "idle" || pipelineStatus === "error" ? (
              <>
                <div className="field">
                  <label htmlFor="token">GitHub Personal Access Token</label>
                  <input id="token" type="password" placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                    value={token} onChange={(e) => setToken(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !cachePrompt && handleSetup()} autoComplete="off" />
                </div>
                {setupError && <div className="error-msg">{setupError}</div>}

                {cachePrompt ? (
                  <div className="cache-prompt">
                    <p className="cache-prompt-msg">
                      Found a search history cache from a previous session. Use it for faster results?
                    </p>
                    <div className="cache-prompt-actions">
                      <button className="btn" onClick={() => _doSetup(true)}>
                        Yes, use cache
                      </button>
                      <button className="btn btn-secondary" onClick={() => _doSetup(false)}>
                        No, start fresh
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: "8px" }}>
                    <button className="btn" onClick={handleSetup} disabled={!token.trim()}>
                      Load My Bookmarks
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="loading-section">
                <div className="spinner" />
                {phase === "fetching" && (
                  <>
                    {fetchStep === "discovering" ? (
                      <>
                        <p className="progress-label">
                          Discovering your starred repositories…
                          {totalRepos > 0 && (
                            <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                              {" "}&mdash; {totalRepos.toLocaleString()} found so far
                            </span>
                          )}
                        </p>
                        <p className="progress-sub">Paginating through GitHub stars</p>
                        <div className="progress-bar-wrap">
                          <div className="progress-bar-indeterminate" />
                        </div>
                      </>
                    ) : (
                      <>
                        <p className="progress-label">
                          Fetching repos… {repoCount.toLocaleString()}&thinsp;/&thinsp;{totalRepos.toLocaleString()}
                        </p>
                        <p className="progress-sub">Reading READMEs from GitHub</p>
                        <div className="progress-bar-wrap">
                          <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                        </div>
                      </>
                    )}
                  </>
                )}
                {phase === "indexing" && (() => {
                  const stepLabels: Record<string, string> = {
                    loading_cache: "Loading cached index…",
                    bm25:          "Building keyword index…",
                    embedding:     "Embedding repositories…",
                    compiling:     "Compiling search graph…",
                  };
                  const stepSubs: Record<string, string> = {
                    loading_cache: "Reading BM25 + vector store from disk",
                    bm25:          "Indexing repo names, descriptions & topics",
                    embedding:     "Generating vector embeddings — this is the slow part",
                    compiling:     "Assembling the LangGraph orchestrator",
                  };
                  const label   = indexStep ? stepLabels[indexStep] ?? "Building search index…" : "Building search index…";
                  const sub     = indexStep ? stepSubs[indexStep]   ?? "" : "";
                  const showBar = indexStep === "embedding" && indexTotal > 0;
                  const pct     = showBar ? Math.round((indexCount / indexTotal) * 100) : 0;
                  return (
                    <>
                      <p className="progress-label">{label}</p>
                      {sub && <p className="progress-sub">{sub}</p>}
                      {showBar ? (
                        <>
                          <p className="progress-sub" style={{ marginTop: "4px" }}>
                            {indexCount.toLocaleString()}&thinsp;/&thinsp;{indexTotal.toLocaleString()} repos
                            &nbsp;&middot;&nbsp;{pct}%
                          </p>
                          <div className="progress-bar-wrap">
                            <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
                          </div>
                        </>
                      ) : (indexStep === "bm25" || indexStep === "compiling" || indexStep === "loading_cache") ? (
                        <div className="progress-bar-wrap">
                          <div className="progress-bar-indeterminate" />
                        </div>
                      ) : null}
                    </>
                  );
                })()}
              </div>
            )}
          </section>
        )}

        {/* ── Ready banner + reset ── */}
        {pipelineStatus === "ready" && (
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
            <span className={badgeClass[pipelineStatus]}>✓ {statusLabel[pipelineStatus]}</span>
            <button className="btn btn-secondary" onClick={handleReset}>Reset</button>
          </div>
        )}

        {/* ── Query card ── */}
        {pipelineStatus === "ready" && (
          <section className="card">
            <h2>Ask a question</h2>
            <form onSubmit={handleQuery}>
              <div className="query-row">
                <input type="text"
                  placeholder="e.g. What Bayesian statistics repos have I starred?"
                  value={question} onChange={(e) => setQuestion(e.target.value)}
                  disabled={isSearching || isRefining} autoFocus />
                <button type="submit" className="btn"
                  disabled={!question.trim() || isSearching || isRefining}>
                  {isSearching ? "Searching…" : "Ask"}
                </button>
              </div>
            </form>
            {queryError && <div className="error-msg">{queryError}</div>}
          </section>
        )}

        {/* ── Query progress indicator ── */}
        {(isSearching || isRefining) && (() => {
          const step  = queryProgress?.step       ?? 0;
          const total = queryProgress?.totalSteps ?? TOTAL_QUERY_STEPS;
          const r     = 22;
          const circ  = 2 * Math.PI * r;
          const offset = circ * (1 - step / total);
          return (
            <div className="results-card">
              <div className="query-progress">
                <div className="query-progress-body">

                  {/* Left — circular ring */}
                  <div className="query-ring-wrap">
                    <svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
                      {/* Outer dashed ring — spins slowly */}
                      <circle cx="32" cy="32" r="29"
                        fill="none" stroke="var(--border)"
                        strokeWidth="1.5" strokeDasharray="5 3.5"
                        className="query-ring-spin" />
                      {/* Inner progress arc */}
                      <circle cx="32" cy="32" r={r}
                        fill="none" stroke="var(--accent)"
                        strokeWidth="3" strokeLinecap="round"
                        strokeDasharray={`${circ}`}
                        strokeDashoffset={`${offset}`}
                        transform="rotate(-90 32 32)"
                        className="query-ring-arc" />
                    </svg>
                    <span className="query-ring-label">
                      {step}&thinsp;/&thinsp;{total}
                    </span>
                  </div>

                  {/* Right — funny message + step label + bar */}
                  <div className="query-progress-text">
                    <div className="query-funny-row">
                      <span className="query-funny-msg" key={funnyMsgIdx}>
                        {FUNNY_MESSAGES[funnyMsgIdx]}
                      </span>
                    </div>
                    <div className="query-progress-header">
                      <span className="query-progress-label">
                        {queryProgress
                          ? `${queryProgress.label}…`
                          : isRefining
                            ? "Refining based on your feedback…"
                            : "Starting up…"}
                      </span>
                    </div>
                    <div className="query-progress-track">
                      <div
                        className="query-progress-fill"
                        style={{
                          width: queryProgress
                            ? `${Math.round((step / total) * 100)}%`
                            : "4%",
                        }}
                      />
                    </div>
                  </div>

                </div>
              </div>
            </div>
          );
        })()}

        {/* ── Streaming answer preview (while generate_answer is running) ── */}
        {streamingAnswer && (isSearching || isRefining) && (
          <div className="results-card streaming-preview">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingAnswer}</ReactMarkdown>
            <span className="streaming-cursor" aria-hidden="true" />
          </div>
        )}

        {/* ── Current session results ── */}
        {currentSession && !isSearching && !isRefining && (() => {
          const { intro, items } = parseAnswerItems(currentSession.answer);
          const hasItems = items.length > 0 && currentSession.results.length > 0;
          return (
            <div ref={resultsRef}>
              <div className="results-card">
                {/* Heading */}
                <div className="results-heading-row">
                  <h2>
                    Results
                    {currentSession.iteration > 0 && (
                      <span className="round-badge">Round {currentSession.iteration}</span>
                    )}
                  </h2>
                </div>
                {canRefine && (
                  <div className="feedback-meta-row">
                    <span className="feedback-meta feedback-question">
                      Give feedback for interactive search
                    </span>
                    <span className="feedback-meta feedback-round">
                      Round {currentSession.iteration + 1} of {MAX_ROUNDS}
                    </span>
                  </div>
                )}

                {/* Optional intro paragraph from the LLM */}
                {intro && (
                  <div className="markdown-body" style={{ marginBottom: "16px" }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{intro}</ReactMarkdown>
                  </div>
                )}

                {/* Per-result cards: LLM text + rating buttons in one row */}
                {hasItems ? (
                  <div className="result-items">
                    {items.map((itemMd, i) => {
                      const repo = currentSession.results[i];
                      if (!repo) return null;
                      return (
                        <div key={repo.repo} className="result-item">
                          <div className="result-item-content markdown-body">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{itemMd}</ReactMarkdown>
                          </div>
                          {canRefine && (
                            <div className="result-item-rating">
                              {RATINGS.map((r) => (
                                <button
                                  key={r}
                                  className={`rating-btn ${ratings[repo.repo] === r ? "rating-btn--active rating-btn--" + r : ""}`}
                                  onClick={() => handleRatingChange(repo.repo, r)}
                                  title={r.charAt(0).toUpperCase() + r.slice(1)}
                                >
                                  {EMOJI[r]}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  /* Fallback: can't parse items, show raw markdown */
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentSession.answer}</ReactMarkdown>
                  </div>
                )}

                {/* Action buttons */}
                {canRefine && (
                  <>
                    {refineError && <div className="error-msg" style={{ marginTop: "12px" }}>{refineError}</div>}
                    <div className="feedback-actions" style={{ marginTop: "16px" }}>
                      <button className="btn" onClick={() => handleSubmitFeedback(false)}>
                        🔄 Refine Results
                      </button>
                      <button className="btn btn-secondary" onClick={() => handleSubmitFeedback(true)}>
                        ✅ I&apos;m Satisfied
                      </button>
                    </div>
                  </>
                )}

                {/* Session done banner */}
                {currentSession.done && (
                  <div className="done-banner" style={{ marginTop: "16px" }}>
                    ✅ Search complete — ask another question above to start a new search.
                  </div>
                )}
              </div>
            </div>
          );
        })()}

        {/* ── History of previous rounds ── */}
        {history.map((round, i) => {
          const { intro: hIntro, items: hItems } = parseAnswerItems(round.answer);
          const hHasItems = hItems.length > 0;
          return (
            <div key={i} className="results-card history-card">
              <h2>
                {round.iteration > 0
                  ? `Previous results — Round ${round.iteration}`
                  : "Previous search"}
              </h2>
              {hIntro && (
                <div className="markdown-body" style={{ marginBottom: "12px" }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{hIntro}</ReactMarkdown>
                </div>
              )}
              {hHasItems ? (
                <div className="result-items">
                  {hItems.map((itemMd, j) => (
                    <div key={j} className="result-item result-item--history">
                      <div className="result-item-content markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{itemMd}</ReactMarkdown>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{round.answer}</ReactMarkdown>
                </div>
              )}
            </div>
          );
        })}

        <footer className="footer">
          AskMyBookmark &mdash; powered by OpenAI + LangGraph + Qdrant
          {githubUsername && <> &mdash; @{githubUsername}</>}
        </footer>
      </main>
    </>
  );
}
