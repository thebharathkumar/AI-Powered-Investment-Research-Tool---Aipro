# AI-Powered Investment Research Tool (AIPro)

A production-grade, LLM-powered investment research platform that automates financial document analysis, insight extraction, and risk assessment using **LangChain**, **LangGraph**, **RAG**, **PostgreSQL**, and **FastAPI**.

---

## Features

- **Agentic RAG Pipeline** — LangGraph `StateGraph` orchestrates retrieve → analyze steps over financial filings
- **High-accuracy retrieval** — ChromaDB vector store with OpenAI embeddings and MMR retrieval
- **PostgreSQL data layer** — SQLAlchemy 2.0 async models with optimized analytical indexes
- **FastAPI REST API** — Three route groups: research queries, document ingestion, financial analysis
- **Statistical analysis** — Returns, volatility, Sharpe ratio, max drawdown, linear trend, outlier detection
- **Feature engineering** — Momentum, volatility, fundamental ratios, YoY growth, hallucination-reduction scoring
- **Multi-model support** — OpenAI GPT-4 and Anthropic Claude via `AnalysisAgent`
- **AWS-ready** — SAM template for VPC, S3, ECR, ECS Fargate, and RDS PostgreSQL
- **Docker** — Dockerfile + docker-compose with PostgreSQL

---

## Project Structure

```
├── app/
│   ├── config.py                    # Pydantic-settings configuration
│   ├── main.py                      # FastAPI application factory
│   ├── api/routes/
│   │   ├── research.py              # /research — query, history, ticker summary
│   │   ├── documents.py             # /documents — list, chunks, ingest
│   │   └── analysis.py             # /analysis — sentiment, risk, report, metrics, statistics
│   ├── agents/
│   │   ├── research_agent.py        # LangGraph agentic research pipeline
│   │   ├── document_agent.py        # Document ingestion & metadata extraction
│   │   └── analysis_agent.py       # Sentiment, risk, report generation
│   ├── rag/
│   │   ├── pipeline.py              # Chunking, ingestion, RetrievalQA chain
│   │   ├── embeddings.py            # OpenAI embeddings wrapper
│   │   └── retriever.py             # Chroma vector store & retriever
│   ├── db/
│   │   ├── models.py                # SQLAlchemy 2.0 ORM models
│   │   ├── queries.py               # Optimized async analytical queries
│   │   └── session.py               # Async session management
│   └── analytics/
│       ├── statistical_analysis.py  # Returns, volatility, Sharpe, drawdown, trend
│       └── feature_engineering.py  # Momentum, fundamentals, hallucination features
├── tests/
│   ├── test_api.py                  # FastAPI endpoint tests
│   ├── test_rag.py                  # RAG pipeline unit tests
│   ├── test_agents.py               # Agent tests (mocked LLM calls)
│   └── test_analytics.py           # Statistical & feature engineering tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── aws/
│   └── template.yaml                # AWS SAM deployment template
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd AI-Powered-Investment-Research-Tool---Aipro
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Run with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/research/query` | Run agentic research query |
| `GET` | `/api/v1/research/history` | Query history (optionally filtered by ticker) |
| `GET` | `/api/v1/research/summary/{ticker}` | Aggregated analysis summary |
| `GET` | `/api/v1/documents/{ticker}` | List documents for a ticker |
| `POST` | `/api/v1/documents/ingest` | Ingest a financial document |
| `POST` | `/api/v1/analysis/sentiment` | Sentiment analysis |
| `POST` | `/api/v1/analysis/risks` | Risk assessment |
| `POST` | `/api/v1/analysis/report` | Generate research report |
| `POST` | `/api/v1/analysis/statistics` | Compute price statistics |
| `POST` | `/api/v1/analysis/hallucination-check` | Score LLM response reliability |
| `POST` | `/api/v1/analysis/metrics/{ticker}` | Save financial metrics |
| `GET` | `/api/v1/analysis/metrics/{ticker}/timeseries` | Retrieve metrics time series |

---

## Environment Variables

See `.env.example` for all configuration options. Key variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for LLM + embeddings) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional, for Claude) |
| `DATABASE_URL` | Async PostgreSQL connection string |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory |
| `CORS_ORIGINS` | JSON list of allowed CORS origins (default `["*"]`) |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 33 tests pass covering analytics, RAG pipeline, agents (mocked), and API endpoints.

---

## AWS Deployment

The SAM template (`aws/template.yaml`) provisions:
- VPC, S3 bucket (versioned, encrypted, private)
- ECR repository with image scanning
- ECS Fargate cluster
- RDS PostgreSQL (Multi-AZ in production)

```bash
sam build && sam deploy --guided
```

---

## Architecture

```
Client → FastAPI → Research/Document/Analysis Routes
              ↓
         LangGraph Agent (retrieve → analyze)
              ↓              ↓
         ChromaDB RAG    OpenAI / Claude LLM
              ↓
         PostgreSQL (documents, chunks, queries, metrics)
```

**Hallucination reduction** is built in via:
- Zero-temperature LLM calls
- Prompt instructions forbidding fabricated data
- `engineer_hallucination_reduction_features()` scoring numeric overlap, hedging phrases, and uncertainty language
