import Head from "next/head";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://localhost:8000";

type PipelineStatus = "idle" | "loading" | "ready" | "error";
type LoadingPhase = "fetching" | "indexing" | null;

interface StatusResponse {
  status: PipelineStatus;
  phase: LoadingPhase;
  repo_count: number;
  total_repos: number;
  error: string | null;
}

interface QueryResult {
  question: string;
  response: string;
}

export default function Home() {
  // ── Setup state ────────────────────────────────────────────────────────────
  const [token, setToken] = useState("");
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>("idle");
  const [phase, setPhase] = useState<LoadingPhase>(null);
  const [repoCount, setRepoCount] = useState(0);
  const [totalRepos, setTotalRepos] = useState(0);
  const [setupError, setSetupError] = useState<string | null>(null);

  // ── Query state ────────────────────────────────────────────────────────────
  const [question, setQuestion] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [results, setResults] = useState<QueryResult[]>([]);

  // ── Polling ────────────────────────────────────────────────────────────────
  // Fetch status from the backend and update state.
  async function fetchStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (!res.ok) return;
      const data: StatusResponse = await res.json();
      setPipelineStatus(data.status);
      setPhase(data.phase);
      setRepoCount(data.repo_count);
      setTotalRepos(data.total_repos);
      if (data.status === "error") {
        setSetupError(data.error ?? "An unknown error occurred.");
      }
    } catch {
      // Backend not yet reachable — silently ignore
    }
  }

  // Check once on mount so a page refresh picks up an in-progress or ready pipeline.
  useEffect(() => {
    fetchStatus();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll every 2 s while loading; React's cleanup automatically stops it
  // the moment pipelineStatus changes away from "loading".
  useEffect(() => {
    if (pipelineStatus !== "loading") return;
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipelineStatus]);

  // ── Handlers ───────────────────────────────────────────────────────────────
  const handleSetup = async () => {
    if (!token.trim()) return;
    setSetupError(null);
    setPipelineStatus("loading");
    setPhase("fetching");
    setRepoCount(0);
    setTotalRepos(0);

    try {
      const res = await fetch(`${API_BASE}/api/setup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_token: token.trim() }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail ?? "Setup failed.");
      }
      // Polling starts automatically via the useEffect watching pipelineStatus
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSetupError(msg);
      setPipelineStatus("error");
    }
  };

  const handleReset = () => {
    // Setting pipelineStatus to "idle" causes the polling useEffect to
    // run its cleanup (clearInterval) automatically.
    setPipelineStatus("idle");
    setPhase(null);
    setRepoCount(0);
    setTotalRepos(0);
    setSetupError(null);
    setToken("");
    setResults([]);
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isQuerying) return;

    setIsQuerying(true);
    setQueryError(null);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail ?? "Query failed.");
      }
      setResults((prev) => [
        { question: question.trim(), response: data.response },
        ...prev,
      ]);
      setQuestion("");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setQueryError(msg);
    } finally {
      setIsQuerying(false);
    }
  };

  // ── Progress helpers ───────────────────────────────────────────────────────
  const progressPct =
    totalRepos > 0 ? Math.round((repoCount / totalRepos) * 100) : 0;

  const statusLabel: Record<PipelineStatus, string> = {
    idle: "Not loaded",
    loading: "Loading…",
    ready: `${totalRepos.toLocaleString()} repos indexed`,
    error: "Error",
  };

  const badgeClass: Record<PipelineStatus, string> = {
    idle: "badge badge-idle",
    loading: "badge badge-loading",
    ready: "badge badge-ready",
    error: "badge badge-error",
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>AskMyBookmark</title>
        <meta
          name="description"
          content="Query your GitHub starred repositories with AI"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="page">
        {/* Header */}
        <header className="header">
          <h1>
            Ask<span>My</span>Bookmark
          </h1>
          <p>
            Query your GitHub starred repositories using natural language &mdash;
            powered by RAG
          </p>
        </header>

        {/* ── Setup card (always visible unless ready) ── */}
        {pipelineStatus !== "ready" && (
          <section className="card">
            <h2>Connect your GitHub account</h2>

            {pipelineStatus === "idle" || pipelineStatus === "error" ? (
              <>
                <div className="field">
                  <label htmlFor="token">GitHub Personal Access Token</label>
                  <input
                    id="token"
                    type="password"
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSetup()}
                    autoComplete="off"
                  />
                </div>
                {setupError && (
                  <div className="error-msg">{setupError}</div>
                )}
                <div style={{ marginTop: "8px" }}>
                  <button
                    className="btn"
                    onClick={handleSetup}
                    disabled={!token.trim()}
                  >
                    Load My Bookmarks
                  </button>
                </div>
              </>
            ) : (
              /* Loading state */
              <div className="loading-section">
                <div className="spinner" />

                {phase === "fetching" && (
                  <>
                    <p className="progress-label">
                      {totalRepos > 0
                        ? `Fetching repos… ${repoCount.toLocaleString()} / ${totalRepos.toLocaleString()}`
                        : "Discovering your starred repositories…"}
                    </p>
                    <p className="progress-sub">
                      Reading READMEs from GitHub
                    </p>
                    {totalRepos > 0 && (
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${progressPct}%` }}
                        />
                      </div>
                    )}
                  </>
                )}

                {phase === "indexing" && (
                  <>
                    <p className="progress-label">
                      Building search index…
                    </p>
                    <p className="progress-sub">
                      Embedding {totalRepos.toLocaleString()} repositories
                      with OpenAI &mdash; this takes a few minutes
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        )}

        {/* ── Ready banner + reset ── */}
        {pipelineStatus === "ready" && (
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "24px",
            }}
          >
            <span className={badgeClass[pipelineStatus]}>
              ✓ {statusLabel[pipelineStatus]}
            </span>
            <button className="btn btn-secondary" onClick={handleReset}>
              Reset
            </button>
          </div>
        )}

        {/* ── Query card ── */}
        {pipelineStatus === "ready" && (
          <section className="card">
            <h2>Ask a question</h2>
            <form onSubmit={handleQuery}>
              <div className="query-row">
                <input
                  type="text"
                  placeholder="e.g. What Bayesian statistics repos have I starred?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={isQuerying}
                  autoFocus
                />
                <button
                  type="submit"
                  className="btn"
                  disabled={!question.trim() || isQuerying}
                >
                  {isQuerying ? "Searching…" : "Ask"}
                </button>
              </div>
            </form>
            {queryError && <div className="error-msg">{queryError}</div>}
          </section>
        )}

        {/* ── Thinking indicator ── */}
        {isQuerying && (
          <div className="results-card">
            <div className="thinking">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
              <span style={{ marginLeft: "4px" }}>Searching your bookmarks…</span>
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {results.map((r, i) => (
          <div key={i} className="results-card">
            <h2>
              Q: {r.question}
            </h2>
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {r.response}
              </ReactMarkdown>
            </div>
          </div>
        ))}

        <footer className="footer">
          AskMyBookmark &mdash; powered by OpenAI + LangGraph + Qdrant
        </footer>
      </main>
    </>
  );
}
