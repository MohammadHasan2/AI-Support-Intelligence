import os
import time
import logging
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Support Intelligence API",
    version="1.0.0",
)

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        dict.fromkeys([
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            *frontend_origins,
        ])
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value


class SearchRequest(QuestionRequest):
    match_count: int = Field(default=5, ge=1, le=10)


def get_supabase_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        raise HTTPException(
            status_code=500,
            detail="Supabase server configuration is missing.",
        )

    return url, key


async def call_edge_function(
    function_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    url, key = get_supabase_config()

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            f"{url}/functions/v1/{function_name}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Supabase returned an unexpected response.")

    return data


async def record_metric(metric: dict[str, Any]) -> None:
    """Write a metric to Supabase without breaking the main request."""
    try:
        url, key = get_supabase_config()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{url}/rest/v1/request_metrics",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=metric,
            )
            response.raise_for_status()

    except Exception as exc:
        # A metrics outage must not turn a successful answer into a failure.
        logger.warning("Could not persist request metric: %s", type(exc).__name__)


def build_metric(
    request_type: str,
    latency_ms: int,
    outcome: str,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    data = data or {}
    usage = data.get("usage") or {}
    results = data.get("results") or []

    retrieved_count = data.get(
        "evidence_count",
        data.get("result_count", len(results)),
    )

    return {
        "request_type": request_type,
        "latency_ms": max(0, latency_ms),
        "retrieved_count": max(0, int(retrieved_count or 0)),
        "evidence_sufficient": (
            data.get("evidence_sufficient")
            if isinstance(data.get("evidence_sufficient"), bool)
            else None
        ),
        "citations_valid": (
            data.get("citations_valid")
            if isinstance(data.get("citations_valid"), bool)
            else None
        ),
        "outcome": outcome,
        "error_code": error_code,
        "model_name": data.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def proxy_request(
    request_type: str,
    function_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        data = await call_edge_function(function_name, payload)

    except httpx.HTTPStatusError as exc:
        latency = int((time.perf_counter() - started) * 1000)
        await record_metric(
            build_metric(
                request_type,
                latency,
                "failed",
                error_code=f"upstream_http_{exc.response.status_code}",
            )
        )
        raise HTTPException(
            status_code=502,
            detail="The Supabase function returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        latency = int((time.perf_counter() - started) * 1000)
        await record_metric(
            build_metric(
                request_type,
                latency,
                "failed",
                error_code="upstream_connection_error",
            )
        )
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Supabase function.",
        ) from exc

    except ValueError as exc:
        latency = int((time.perf_counter() - started) * 1000)
        await record_metric(
            build_metric(
                request_type,
                latency,
                "failed",
                error_code="invalid_upstream_response",
            )
        )
        raise HTTPException(
            status_code=502,
            detail="The Supabase function returned an invalid response.",
        ) from exc

    latency = int((time.perf_counter() - started) * 1000)

    await record_metric(
        build_metric(
            request_type,
            latency,
            "success",
            data=data,
        )
    )

    return data


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(body: QuestionRequest) -> dict[str, Any]:
    return await proxy_request(
        "chat",
        "answer",
        {"question": body.question.strip()},
    )


@app.post("/api/search")
async def search(body: SearchRequest) -> dict[str, Any]:
    return await proxy_request(
        "search",
        "search",
        {
            "question": body.question.strip(),
            "match_count": body.match_count,
        },
    )

@app.get("/api/evaluations")
async def get_evaluations() -> dict[str, Any]:
    """Return recent evaluation runs and their per-question results."""
    url, key = get_supabase_config()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            runs_response = await client.get(
                f"{url}/rest/v1/evaluation_runs",
                headers=headers,
                params={
                    "select": (
                        "id,run_name,config,dataset_size,metrics,"
                        "failure_examples,status,started_at,completed_at"
                    ),
                    "order": "started_at.desc",
                    "limit": "20",
                },
            )
            runs_response.raise_for_status()
            runs = runs_response.json()

            if not isinstance(runs, list):
                raise ValueError("Invalid evaluation runs response.")

            if not runs:
                return {"runs": []}

            run_ids = [
                str(run["id"])
                for run in runs
                if run.get("id")
            ]

            if not run_ids:
                return {"runs": runs}

            results_response = await client.get(
                f"{url}/rest/v1/evaluation_results",
                headers=headers,
                params={
                    "select": (
                        "id,run_id,question_id,expected_sources,"
                        "retrieved_sources,metrics,failure_reason"
                    ),
                    "run_id": f"in.({','.join(run_ids)})",
                    "limit": "1000",
                },
            )
            results_response.raise_for_status()
            results = results_response.json()

            if not isinstance(results, list):
                raise ValueError("Invalid evaluation results response.")

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Evaluation query failed with HTTP %s",
                exc.response.status_code,
            )
            raise HTTPException(
                status_code=502,
                detail="Could not load evaluation results from Supabase.",
            ) from exc

        except httpx.RequestError as exc:
            logger.warning(
                "Evaluation query connection failed: %s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Could not connect to the evaluation database.",
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Supabase returned invalid evaluation data.",
            ) from exc

    results_by_run: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        run_id = str(result.get("run_id", ""))
        results_by_run.setdefault(run_id, []).append(result)

    for run in runs:
        run["results"] = results_by_run.get(str(run.get("id", "")), [])

    return {"runs": runs}