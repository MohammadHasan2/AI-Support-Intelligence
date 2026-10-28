import json
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_BASE = os.getenv(
    "EVAL_API_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
K = 5

with (ROOT / "evaluation" / "dataset.json").open(
    encoding="utf-8-sig"
) as file:
    CASES = json.load(file)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value):
    value = str(value or "").lower()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def get_payload(response):
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def get_results(payload):
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def get_chunk_id(result):
    return str(result.get("chunk_id", ""))


def get_source(result):
    source = result.get("source", {})
    if isinstance(source, dict):
        value = source.get("source_path") or source.get("source_name")
    else:
        value = source
    if not value:
        value = result.get("source_path") or result.get("source_name")
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()


def average(values):
    return round(statistics.mean(values), 4) if values else None


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile.
    index = max(0, math.ceil(percentile_value / 100 * len(ordered)) - 1)
    return round(ordered[index], 2)


def retrieval_metrics(expected_ids, results, k=K):
    ranked_ids = [get_chunk_id(item) for item in results[:k]]
    relevant = set(expected_ids)
    hits = [1 if chunk_id in relevant else 0 for chunk_id in ranked_ids]

    relevant_retrieved = sum(hits)
    precision = relevant_retrieved / k
    recall = relevant_retrieved / len(relevant) if relevant else None

    first_rank = next(
        (rank for rank, hit in enumerate(hits, start=1) if hit), None
    )
    reciprocal_rank = 1 / first_rank if first_rank else 0.0

    dcg = sum(
        hit / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    ideal_relevant = min(len(relevant), k)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    ndcg = dcg / idcg if idcg else None

    return {
        "k": k,
        "retrieved_chunk_ids": ranked_ids,
        "relevant_chunks_found": relevant_retrieved,
        "relevant_chunk_count": len(relevant),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4) if recall is not None else None,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "ndcg_at_k": round(ndcg, 4) if ndcg is not None else None,
        "first_relevant_rank": first_rank,
    }


def required_fact_coverage(answer, required_facts):
    if not required_facts:
        return None, []

    normalized_answer = normalize_text(answer)
    missing = [
        fact for fact in required_facts
        if normalize_text(fact) not in normalized_answer
    ]
    matched = len(required_facts) - len(missing)

    return round(matched / len(required_facts), 4), missing


def refusal_detected(answer):
    text = normalize_text(answer)
    phrases = [
        "i do not have enough information",
        "i don't have enough information",
        "insufficient evidence",
        "cannot determine from the available information",
        "can't determine from the available information",
        "not available in the knowledge base",
        "i cannot verify",
        "i don't know",
    ]
    return any(normalize_text(phrase) in text for phrase in phrases)


def supabase_request(client, method, table, **kwargs):
    response = client.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{table}",
        **kwargs,
    )
    response.raise_for_status()
    return response


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in root .env"
        )

    run_id = str(uuid.uuid4())
    run_name = f"support-rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    run_payload = {
        "id": run_id,
        "run_name": run_name,
        "config": {
            "dataset": "dataset.json",
            "k": K,
            "api_base": API_BASE,
            "retrieval_metric": "exact_chunk_id",
            "fact_metric": "normalized_required_fact_coverage",
        },
        "dataset_size": len(CASES),
        "metrics": {},
        "failure_examples": [],
        "status": "running",
        "started_at": utc_now(),
    }

    result_rows = []
    failures = []
    retrieval_times = []
    answer_times = []
    retrieval_scores = []
    source_hits = []
    fact_scores = []
    citation_scores = []
    abstention_scores = []
    answerable_cases = 0
    unanswerable_cases = 0
    search_successes = 0
    answer_successes = 0

    with httpx.Client(timeout=60.0) as api, httpx.Client(
        headers=headers, timeout=30.0
    ) as db:
        supabase_request(db, "POST", "evaluation_runs", json=run_payload)

        try:
            for case in CASES:
                case_id = case["id"]
                answerable = bool(case["answerable"])
                expected_ids = case["relevant_chunk_ids"]
                expected_sources = [
                    source.lower() for source in case["expected_sources"]
                ]
                row_failure = None
                metrics = {
                    "answerable": answerable,
                    "retrieval_success": False,
                    "answer_success": False,
                }
                retrieved_results = []
                retrieved_sources = []
                answer = ""
                evidence_count = 0

                if answerable:
                    answerable_cases += 1
                else:
                    unanswerable_cases += 1

                # 1. Retrieval evaluation
                start = time.perf_counter()
                try:
                    response = api.post(
                        f"{API_BASE}/api/search",
                        json={"question": case["question"], "match_count": K},
                    )
                    retrieval_ms = round(
                        (time.perf_counter() - start) * 1000, 2
                    )
                    retrieval_times.append(retrieval_ms)
                    response.raise_for_status()
                    search_payload = get_payload(response)
                    retrieved_results = get_results(search_payload)
                    retrieved_sources = list(dict.fromkeys(
                        source for source in
                        (get_source(item) for item in retrieved_results)
                        if source
                    ))
                    metrics["retrieval_success"] = True
                    metrics["retrieval_latency_ms"] = retrieval_ms
                    search_successes += 1

                    if answerable:
                        scored = retrieval_metrics(expected_ids, retrieved_results)
                        metrics.update(scored)
                        metrics["source_hit"] = int(
                            any(s in retrieved_sources for s in expected_sources)
                        )
                        source_hits.append(metrics["source_hit"])
                        retrieval_scores.append(scored)
                except Exception as exc:
                    retrieval_ms = round(
                        (time.perf_counter() - start) * 1000, 2
                    )
                    metrics["retrieval_latency_ms"] = retrieval_ms
                    row_failure = f"Search: {type(exc).__name__}: {exc}"
                    failures.append({
                        "question_id": case_id,
                        "reason": row_failure,
                    })

                # 2. Answer evaluation
                start = time.perf_counter()
                try:
                    response = api.post(
                        f"{API_BASE}/api/chat",
                        json={"question": case["question"]},
                    )
                    answer_ms = round(
                        (time.perf_counter() - start) * 1000, 2
                    )
                    answer_times.append(answer_ms)
                    response.raise_for_status()
                    payload = get_payload(response)
                    answer = str(payload.get("answer", ""))
                    evidence_count = int(
                        payload.get("evidence_count", len(retrieved_results)) or 0
                    )
                    metrics["answer_success"] = True
                    metrics["answer_latency_ms"] = answer_ms
                    answer_successes += 1

                    citation_ids = [
                        int(n) for n in re.findall(r"\[S(\d+)\]", answer)
                    ]
                    citation_ids_valid = bool(citation_ids) and all(
                        1 <= n <= evidence_count for n in citation_ids
                    )
                    metrics["citation_ids"] = citation_ids
                    metrics["citation_ids_valid"] = citation_ids_valid
                    metrics["citation_present"] = bool(citation_ids)

                    if answerable:
                        citation_scores.append(int(citation_ids_valid))
                        coverage, missing_facts = required_fact_coverage(
                            answer, case["required_facts"]
                        )
                        metrics["required_fact_coverage"] = coverage
                        metrics["missing_required_facts"] = missing_facts
                        fact_scores.append(coverage)
                    else:
                        refused = refusal_detected(answer)
                        metrics["refusal_detected"] = refused

                    # Accuracy of the decision to answer versus abstain.
                    if answerable:
                        abstention_scores.append(int(not refusal_detected(answer)))
                    else:
                        abstention_scores.append(int(refusal_detected(answer)))

                except Exception as exc:
                    answer_ms = round(
                        (time.perf_counter() - start) * 1000, 2
                    )
                    metrics["answer_latency_ms"] = answer_ms
                    reason = f"Answer: {type(exc).__name__}: {exc}"
                    row_failure = (
                        f"{row_failure}; {reason}" if row_failure else reason
                    )
                    failures.append({
                        "question_id": case_id,
                        "reason": reason,
                    })

                result_rows.append({
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "question_id": case_id,
                    "expected_sources": case["expected_sources"],
                    "retrieved_sources": retrieved_sources,
                    "metrics": metrics,
                    "failure_reason": row_failure,
                })

                print(
                    f"{case_id}: retrieval={metrics['retrieval_success']}, "
                    f"answer={metrics['answer_success']}, "
                    f"source_hit={metrics.get('source_hit', 'N/A')}, "
                    f"failure={row_failure or 'none'}"
                )

            # Persist individual results.
            for row in result_rows:
                supabase_request(
                    db, "POST", "evaluation_results", json=row
                )

            # Aggregate retrieval metrics only across answerable cases.
            def score_avg(key):
                values = [
                    row[key] for row in retrieval_scores
                    if row.get(key) is not None
                ]
                return average(values)

            aggregate = {
                "dataset_size": len(CASES),
                "answerable_questions": answerable_cases,
                "unanswerable_questions": unanswerable_cases,
                "search_success_rate": round(
                    search_successes / len(CASES), 4
                ),
                "answer_success_rate": round(
                    answer_successes / len(CASES), 4
                ),
                "source_hit_rate_at_k": average(source_hits),
                "precision_at_k": score_avg("precision_at_k"),
                "recall_at_k": score_avg("recall_at_k"),
                "mrr": score_avg("reciprocal_rank"),
                "ndcg_at_k": score_avg("ndcg_at_k"),
                "required_fact_coverage": average(fact_scores),
                "citation_id_validity_rate": average(citation_scores),
                "answerability_decision_accuracy": average(abstention_scores),
                "retrieval_latency_p50_ms": percentile(retrieval_times, 50),
                "retrieval_latency_p95_ms": percentile(retrieval_times, 95),
                "answer_latency_p50_ms": percentile(answer_times, 50),
                "answer_latency_p95_ms": percentile(answer_times, 95),
                "failure_count": len(failures),
                "k": K,
            }

            supabase_request(
                db,
                "PATCH",
                "evaluation_runs",
                params={"id": f"eq.{run_id}"},
                json={
                    "status": "completed",
                    "metrics": aggregate,
                    "failure_examples": failures[:10],
                    "completed_at": utc_now(),
                },
            )

            print("\nAggregate metrics:")
            print(json.dumps(aggregate, indent=2))
            print(f"\nSaved evaluation run: {run_id}")

        except Exception as exc:
            try:
                supabase_request(
                    db,
                    "PATCH",
                    "evaluation_runs",
                    params={"id": f"eq.{run_id}"},
                    json={
                        "status": "failed",
                        "metrics": {"error": f"{type(exc).__name__}: {exc}"},
                        "failure_examples": failures[:10],
                        "completed_at": utc_now(),
                    },
                )
            except Exception as update_exc:
                print(f"Could not update failed run: {update_exc}")
            raise


if __name__ == "__main__":
    main()