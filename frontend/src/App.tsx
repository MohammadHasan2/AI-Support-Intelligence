import { useEffect, useState, type FormEvent } from "react";
import "./App.css";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

type Evidence = {
  id?: string;
  source?:
    | string
    | {
        source_path?: string;
        source_name?: string;
      };
  source_name?: string;
  source_path?: string;
  content?: string;
  snippet?: string;
  similarity?: number;
};

type ApiResult = {
  answer?: string;
  citations?: Evidence[];
  results?: Evidence[];
  evidence_count?: number;
  result_count?: number;
  model?: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
};

type Message = {
  role: "user" | "assistant";
  text: string;
  data?: ApiResult;
};

type EvaluationResult = {
  id: string;
  run_id: string;
  question_id: string;
  expected_sources: unknown;
  retrieved_sources: unknown;
  metrics: Record<string, unknown> | null;
  failure_reason: string | null;
};

type EvaluationRun = {
  id: string;
  run_name: string;
  config: Record<string, unknown> | null;
  dataset_size: number;
  metrics: Record<string, unknown> | null;
  failure_examples: unknown;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  results: EvaluationResult[];
};

type EvaluationResponse = {
  runs: EvaluationRun[];
};

function metricNumber(
  metrics: Record<string, unknown> | null,
  key: string,
): number | null {
  const value = metrics?.[key];

  return typeof value === "number" && Number.isFinite(value)
    ? value
    : null;
}

function percentage(value: number | null): string {
  return value === null
    ? "—"
    : `${(value * 100).toFixed(2)}%`;
}

function decimal(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

function latency(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(0)} ms`;
}

function displaySources(value: unknown): string {
  if (value == null) return "—";

  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const source = item as Record<string, unknown>;
          return String(
            source.source_name ??
              source.source_path ??
              source.source ??
              source.id ??
              "Source",
          );
        }
        return String(item);
      })
      .join(", ") || "—";
  }

  if (typeof value === "string") return value;

  return JSON.stringify(value);
}

function resultStatus(result: EvaluationResult): string {
  if (result.failure_reason) return "Execution failure";

  const metrics = result.metrics ?? {};

  if (metrics.execution_success === false) {
    return "Execution failure";
  }

  if (metrics.passed === false || metrics.answer_success === false) {
    return "Quality failure";
  }

  if (metrics.passed === true || metrics.answer_success === true) {
    return "Passed";
  }

  return "Recorded";
}

async function requestApi(
  endpoint: string,
  body: Record<string, unknown>,
): Promise<ApiResult> {
  const response = await fetch(`${API}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    const detail =
      typeof error?.detail === "string"
        ? error.detail
        : `Request failed (${response.status})`;

    throw new Error(detail);
  }

  return response.json();
}

function sourceName(item: Evidence, index: number): string {
  if (typeof item.source === "string") {
    return item.source;
  }

  return (
    item.source?.source_name ??
    item.source?.source_path ??
    item.source_name ??
    item.source_path ??
    `Source ${index + 1}`
  );
}

function EvidenceList({ items }: { items: Evidence[] }) {
  if (!items.length) {
    return <p className="muted">No source details were returned.</p>;
  }

  return (
    <div className="evidence-list">
      {items.map((item, index) => (
        <article
          className="evidence-card"
          key={`${item.id ?? "source"}-${index}`}
        >
          <div className="evidence-heading">
            <span className="source-tag">
              {item.id ?? `S${index + 1}`}
            </span>

            <strong>{sourceName(item, index)}</strong>

            {typeof item.similarity === "number" && (
              <span className="score">
                {(item.similarity * 100).toFixed(1)}% match
              </span>
            )}
          </div>

          {(item.content ?? item.snippet) && (
            <p>{item.content ?? item.snippet}</p>
          )}
        </article>
      ))}
    </div>
  );
}

function EvaluationDashboard() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");

  async function loadEvaluations() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/api/evaluations`);

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : `Could not load evaluations (${response.status}).`,
        );
      }

      const result: EvaluationResponse = await response.json();
      setData(result);

      setSelectedRunId((current) =>
        result.runs.some((run) => run.id === current)
          ? current
          : result.runs[0]?.id ?? "",
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadEvaluations();
  }, []);

  const runs = data?.runs ?? [];
  const selectedRun = runs.find((run) => run.id === selectedRunId);
  const metrics = selectedRun?.metrics ?? {};
  const results = selectedRun?.results ?? [];

  if (loading) {
    return (
      <section className="evaluation-panel">
        <div className="loading-state">
          <span className="spinner" />
          Loading evaluation results…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="evaluation-panel">
        <div className="error-banner" role="alert">
          <strong>Evaluation data unavailable</strong>
          <span>{error}</span>
        </div>
        <button className="refresh-button" onClick={() => void loadEvaluations()}>
          Retry
        </button>
      </section>
    );
  }

  if (!runs.length || !selectedRun) {
    return (
      <section className="evaluation-panel">
        <div className="evaluation-heading">
          <div>
            <div className="eyebrow">RAG QUALITY & PERFORMANCE</div>
            <h2>Evaluation dashboard</h2>
          </div>
          <button
            className="refresh-button"
            onClick={() => void loadEvaluations()}
          >
            Refresh
          </button>
        </div>
        <div className="evaluation-empty">
          <h3>No evaluation runs found</h3>
          <p>
            Run the benchmark and save its results to Supabase to populate
            this dashboard.
          </p>
        </div>
      </section>
    );
  }

  const cards = [
    {
      label: "Source Hit Rate@5",
      value: percentage(metricNumber(metrics, "source_hit_rate_at_k")),
      description: "Questions with an expected source retrieved",
    },
    {
      label: "Required-Fact Coverage",
      value: percentage(metricNumber(metrics, "required_fact_coverage")),
      description: "Required facts detected by the benchmark",
    },
    {
      label: "Citation-ID Validity",
      value: percentage(metricNumber(metrics, "citation_id_validity_rate")),
      description: "Citation IDs that match returned evidence",
    },
    {
      label: "Answerability Accuracy",
      value: percentage(
        metricNumber(metrics, "answerability_decision_accuracy"),
      ),
      description: "Answerable vs. unanswerable decisions",
    },
    {
      label: "Precision@5",
      value: percentage(metricNumber(metrics, "precision_at_k")),
      description: "Relevant results among the top five",
    },
    {
      label: "Recall@5",
      value: percentage(metricNumber(metrics, "recall_at_k")),
      description: "Expected relevant sources retrieved",
    },
    {
      label: "MRR",
      value: decimal(metricNumber(metrics, "mrr")),
      description: "Rank of the first relevant result",
    },
    {
      label: "nDCG@5",
      value: decimal(metricNumber(metrics, "ndcg_at_k")),
      description: "Ranking quality, normalized",
    },
    {
      label: "Retrieval latency P50",
      value: latency(metricNumber(metrics, "retrieval_latency_p50_ms")),
      description: "Median search latency",
    },
    {
      label: "Retrieval latency P95",
      value: latency(metricNumber(metrics, "retrieval_latency_p95_ms")),
      description: "95th-percentile search latency",
    },
    {
      label: "Answer latency P50",
      value: latency(metricNumber(metrics, "answer_latency_p50_ms")),
      description: "Median end-to-end answer latency",
    },
    {
      label: "Answer latency P95",
      value: latency(metricNumber(metrics, "answer_latency_p95_ms")),
      description: "95th-percentile answer latency",
    },
  ];

  const executionFailures = results.filter(
    (result) => result.failure_reason || resultStatus(result) === "Execution failure",
  ).length;

  return (
    <section className="evaluation-panel">
      <div className="evaluation-heading">
        <div>
          <div className="eyebrow">RAG QUALITY & PERFORMANCE</div>
          <h2>Evaluation dashboard</h2>
          <p>
            Inspect retrieval quality, answer checks, and measured latency
            across your benchmark runs.
          </p>
        </div>
        <button
          className="refresh-button"
          onClick={() => void loadEvaluations()}
        >
          Refresh data
        </button>
      </div>

      <div className="run-selector">
        <label htmlFor="evaluation-run">Evaluation run</label>
        <select
          id="evaluation-run"
          value={selectedRunId}
          onChange={(event) => setSelectedRunId(event.target.value)}
        >
          {runs.map((run) => (
            <option key={run.id} value={run.id}>
              {run.run_name} · {run.status} · {run.dataset_size} cases
            </option>
          ))}
        </select>
        <span className="run-date">
          {selectedRun.started_at
            ? new Date(selectedRun.started_at).toLocaleString()
            : "Date unavailable"}
        </span>
      </div>

      <div className="evaluation-summary">
        <div>
          <span>Dataset size</span>
          <strong>{selectedRun.dataset_size}</strong>
        </div>
        <div>
          <span>Saved case results</span>
          <strong>{results.length}</strong>
        </div>
        <div>
          <span>Recorded execution failures</span>
          <strong>{executionFailures}</strong>
        </div>
        <div>
          <span>Run status</span>
          <strong>{selectedRun.status || "Unknown"}</strong>
        </div>
      </div>

      <div className="evaluation-grid">
        {cards.map((card) => (
          <article className="evaluation-card" key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.description}</p>
          </article>
        ))}
      </div>

      <div className="evaluation-section-heading">
        <div>
          <h3>Benchmark question results</h3>
          <p>
            Inspect the stored expected sources, retrieved sources, metrics,
            and failure reasons for each test case.
          </p>
        </div>
      </div>

      {results.length ? (
        <div className="evaluation-table-wrap">
          <table className="evaluation-table">
            <thead>
              <tr>
                <th>Question ID</th>
                <th>Status</th>
                <th>Expected sources</th>
                <th>Retrieved sources</th>
                <th>Metrics / failure details</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => (
                <tr key={result.id}>
                  <td>{result.question_id}</td>
                  <td>
                    <span
                      className={`evaluation-status ${
                        resultStatus(result)
                          .toLowerCase()
                          .replaceAll(" ", "-")
                      }`}
                    >
                      {resultStatus(result)}
                    </span>
                  </td>
                  <td>{displaySources(result.expected_sources)}</td>
                  <td>{displaySources(result.retrieved_sources)}</td>
                  <td>
                    {result.failure_reason ? (
                      <span className="failure-detail">
                        {result.failure_reason}
                      </span>
                    ) : (
                      <code>
                        {JSON.stringify(result.metrics ?? {})}
                      </code>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="evaluation-empty">
          No per-question results were saved for this run.
        </div>
      )}

      <div className="evaluation-note">
        <strong>How to interpret these metrics</strong>
        <p>
          Citation-ID validity checks that citation references exist; it does
          not establish that every claim is supported. Precision and recall
          depend on the benchmark's expected-source labels. Latency includes
          the environment and external services used during the run.
        </p>
        {executionFailures > 0 && (
          <p>
            This run includes {executionFailures} recorded execution failure(s).
            Inspect the failure details before interpreting answer-success
            results as a measure of answer quality.
          </p>
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [mode, setMode] = useState<"chat" | "search" | "evaluation">("chat");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [searchData, setSearchData] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();

    if (!text || loading) return;

    setError("");
    setLoading(true);
    setQuestion("");

    if (mode === "chat") {
      setMessages((previous) => [
        ...previous,
        { role: "user", text },
      ]);
    } else {
      setSearchData(null);
    }

    try {
      const data =
        mode === "chat"
          ? await requestApi("/api/chat", { question: text })
          : await requestApi("/api/search", {
              question: text,
              match_count: 5,
            });

      if (mode === "chat") {
        setMessages((previous) => [
          ...previous,
          {
            role: "assistant",
            text: data.answer ?? "No answer text was returned.",
            data,
          },
        ]);
      } else {
        setSearchData(data);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">S</div>
          <div>
            <strong>Support Intelligence</strong>
            <span>Knowledge assistant</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <button
          className={`nav-item ${mode === "chat" ? "active" : ""}`}
          onClick={() => {
            setMode("chat");
            setError("");
          }}
        >
          <span>✳</span> AI Assistant
        </button>

        <button
          className={`nav-item ${mode === "search" ? "active" : ""}`}
          onClick={() => {
            setMode("search");
            setError("");
          }}
        >
          <span>⌕</span> Retrieval Explorer
        </button>

        <button
          className={`nav-item ${mode === "evaluation" ? "active" : ""}`}
          onClick={() => {
            setMode("evaluation");
            setError("");
          }}
        >
          <span>▥</span> Evaluation Dashboard
        </button>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>RAG pipeline interface</span>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>
              {mode === "chat"
                ? "AI Assistant"
                : mode === "search"
                  ? "Retrieval Explorer"
                  : "Evaluation Dashboard"}
            </strong>
          </div>
        </header>

        <section className="workspace">
          <div className="intro">
            <div className="eyebrow">AI-POWERED KNOWLEDGE RETRIEVAL</div>
            <h1>
              {mode === "chat"
                ? "Answers grounded in your knowledge."
                : mode === "search"
                  ? "Explore your retrieved evidence."
                  : "Measure the quality of your RAG pipeline."}
            </h1>
            <p>
              {mode === "chat"
                ? "Ask a support question. Get an answer with traceable source citations."
                : mode === "search"
                  ? "See which knowledge-base chunks match your question."
                  : "Review benchmark metrics, retrieval performance, citations, and failures."}
            </p>
          </div>
          {mode === "evaluation" && <EvaluationDashboard />}
          {mode === "chat" && (
            <div className="conversation">
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon">✳</div>
                  <h2>What would you like to know?</h2>
                  <p>
                    Try one of these questions to explore your knowledge base.
                  </p>

                  <div className="suggestions">
                    {[
                      "How long does a customer have to request a refund?",
                      "How can I reset a forgotten password?",
                      "What is the shipping policy?",
                    ].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => setQuestion(suggestion)}
                      >
                        {suggestion} <span>↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message, index) => (
                <article
                  className={`message ${message.role}`}
                  key={index}
                >
                  <div className="message-avatar">
                    {message.role === "user" ? "Y" : "S"}
                  </div>

                  <div className="message-body">
                    <div className="message-author">
                      {message.role === "user" ? "You" : "AI Assistant"}
                    </div>

                    <p className="answer-text">{message.text}</p>

                    {message.data && (
                      <>
                        {!!message.data.citations?.length && (
                          <div className="citations">
                            <h3>Sources cited</h3>
                            <EvidenceList items={message.data.citations} />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="loading-state">
                  <span className="spinner" />
                  {mode === "chat"
                    ? "Searching evidence and generating an answer…"
                    : "Retrieving relevant chunks…"}
                </div>
              )}
            </div>
          )}

          {mode === "search" && searchData && (
            <section className="results-panel">
              <div className="results-header">
                <h2>Retrieved chunks</h2>
                <span>
                  {searchData.result_count ?? searchData.results?.length ?? 0}{" "}
                  results
                </span>
              </div>

              <EvidenceList items={searchData.results ?? []} />
            </section>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <strong>Request failed</strong>
              <span>{error}</span>
            </div>
          )}

          {mode !== "evaluation" && (
            <div className="composer-wrap">
              <form className="composer" onSubmit={submitQuestion}>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder={
                    mode === "chat"
                      ? "Ask a question about your support knowledge base…"
                      : "Search your knowledge base…"
                  }
                  maxLength={4000}
                  rows={2}
                  disabled={loading}
                />
              <div className="composer-footer">
                <span>Answers are based on retrieved knowledge</span>
                <button
                  className="send-button"
                  type="submit"
                  disabled={loading || !question.trim()}
                >
                  {loading ? "Working…" : "Submit"} <span>↑</span>
                </button>
              </div>
            </form>

            <p className="disclaimer">
              Verify important answers against the cited source documents.
            </p>
          </div>
        )}
        </section>
      </section>
    </main>
  );
}
