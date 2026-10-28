# AI Support Intelligence

A grounded customer-support RAG application with structured
ticket classification, reproducible retrieval evaluation,
source citations, and runtime observability.

## Project status

Phase 1: feasibility and setup in progress.

No evaluation results or deployment claims are published until
they have been verified.

## Architecture

- Frontend: React + Vite
- API: FastAPI; Cloudflare Workers compatibility under verification
- Database: Supabase PostgreSQL + pgvector
- Embeddings: local Sentence Transformers for ingestion
- LLM: Groq API
- Evaluation: local Python + pytest

## Configuration

Copy .env.example to .env and populate only the credentials
required for the component being tested.

Never commit .env or expose service-role credentials in frontend code.
