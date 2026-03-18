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

interface StatusResponse {
  status:      PipelineStatus;
  phase:       LoadingPhase;
  repo_count:  number;
  total_repos: number;
  error:       string | null;
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

  // ── Query state ────────────────────────────────────────────────────────────
  const [question, setQuestion]         = useState("");
  const [isSearching, setIsSearching]   = useState(false);
  const [queryError, setQueryError]     = useState<string | null>(null);

  // ── Session / feedback state ───────────────────────────────────────────────
  const [currentSession, setCurrentSession]   = useState<SessionResponse | null>(null);
  const [ratings, setRatings]                 = useState<Record<string, EmojiRating>>({});
  const [isRefining, setIsRefining]           = useState(false);
  const [refineError, setRefineError]         = useState<string | null>(null);

  // History of completed rounds (shown above current)
  const [history, setHistory]                 = useState<SearchRound[]>([]);

  const resultsRef = useRef<HTMLDivElement>(null);

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
    setPipelineStatus("loading");
    setPhase("fetching");
    setRepoCount(0);
    setTotalRepos(0);
    try {
      const res  = await fetch(`${API_BASE}/api/setup`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_token: token.trim() }),
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
    setSetupError(null); setToken("");
    setCurrentSession(null); setHistory([]); setRatings({});
    setQuestion("");
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isSearching) return;

    // If there's an open session, archive it first
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

    try {
      const res  = await fetch(`${API_BASE}/api/session/start`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), top_k: 10 }),
      });
      const data: SessionResponse = await res.json();
      if (!res.ok) throw new Error((data as any).detail ?? "Search failed.");
      setCurrentSession(data);
      // Default all ratings to meh
      const defaultRatings: Record<string, EmojiRating> = {};
      data.results.forEach((r) => { defaultRatings[r.repo] = "meh"; });
      setRatings(defaultRatings);
      setQuestion("");
    } catch (err: unknown) {
      setQueryError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSearching(false);
    }
  };

  const handleRatingChange = (repo: string, rating: EmojiRating) => {
    setRatings((prev) => ({ ...prev, [repo]: rating }));
  };

  const handleSubmitFeedback = async (done: boolean) => {
    if (!currentSession) return;
    setIsRefining(true);
    setRefineError(null);

    try {
      const res  = await fetch(`${API_BASE}/api/session/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSession.session_id,
          ratings,
          done,
        }),
      });
      const data: SessionResponse = await res.json();
      if (!res.ok) throw new Error((data as any).detail ?? "Feedback failed.");

      // Push the current round to history before updating
      setHistory((h) => [
        { question: "",
          answer: currentSession.answer, results: currentSession.results,
          iteration: currentSession.iteration, sessionId: null },
        ...h,
      ]);

      if (data.done) {
        // Session ended — show final results without feedback panel
        setCurrentSession({ ...data, done: true });
        setRatings({});
      } else {
        setCurrentSession(data);
        const newRatings: Record<string, EmojiRating> = {};
        data.results.forEach((r) => { newRatings[r.repo] = "meh"; });
        setRatings(newRatings);
      }
    } catch (err: unknown) {
      setRefineError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsRefining(false);
    }
  };

  // ── Derived state ──────────────────────────────────────────────────────────
  const progressPct = totalRepos > 0 ? Math.round((repoCount / totalRepos) * 100) : 0;

  const statusLabel: Record<PipelineStatus, string> = {
    idle:    "Not loaded",
    loading: "Loading…",
    ready:   `${totalRepos.toLocaleString()} repos indexed`,
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
          <p>Query your GitHub starred repositories using natural language &mdash; powered by RAG</p>
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
                    onKeyDown={(e) => e.key === "Enter" && handleSetup()} autoComplete="off" />
                </div>
                {setupError && <div className="error-msg">{setupError}</div>}
                <div style={{ marginTop: "8px" }}>
                  <button className="btn" onClick={handleSetup} disabled={!token.trim()}>
                    Load My Bookmarks
                  </button>
                </div>
              </>
            ) : (
              <div className="loading-section">
                <div className="spinner" />
                {phase === "fetching" && (
                  <>
                    <p className="progress-label">
                      {totalRepos > 0
                        ? `Fetching repos… ${repoCount.toLocaleString()} / ${totalRepos.toLocaleString()}`
                        : "Discovering your starred repositories…"}
                    </p>
                    <p className="progress-sub">Reading READMEs from GitHub</p>
                    {totalRepos > 0 && (
                      <div className="progress-bar-wrap">
                        <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                      </div>
                    )}
                  </>
                )}
                {phase === "indexing" && (
                  <>
                    <p className="progress-label">Building search index…</p>
                    <p className="progress-sub">
                      Embedding {totalRepos.toLocaleString()} repositories with OpenAI &mdash; this takes a few minutes
                    </p>
                  </>
                )}
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

        {/* ── Thinking indicator ── */}
        {(isSearching || isRefining) && (
          <div className="results-card">
            <div className="thinking">
              <span className="dot" /><span className="dot" /><span className="dot" />
              <span style={{ marginLeft: "4px" }}>
                {isRefining ? "Refining based on your feedback…" : "Searching your bookmarks…"}
              </span>
            </div>
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
                  {canRefine && (
                    <span className="feedback-meta">
                      Rate each result · Round {currentSession.iteration + 1} of {MAX_ROUNDS}
                    </span>
                  )}
                </div>

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
        </footer>
      </main>
    </>
  );
}
