"""
main.py — FastAPI application entry point for Verity API.

This file:
  1. Creates the FastAPI application instance with metadata.
  2. Registers CORS middleware so the React frontend can call the API during
     local development.
  3. Includes the evaluation router under the /api/v1 prefix.
  4. Creates all database tables on startup (via SQLAlchemy).

Run with:
  uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.retrieval import router as retrieval_router
from app.core.database import Base, engine


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code that runs once when the server starts (before serving requests).

    We create all SQLAlchemy-managed tables here. `checkfirst=True` (the
    default inside create_all) means existing tables are not re-created,
    so the database is never wiped on restart.
    """
    Base.metadata.create_all(bind=engine)
    yield
    # Any teardown code would go here (e.g., close connection pools)


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Verity API",
    description="Multi-Agent AI Response Evaluation & Hallucination Detection Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# Allow the Vite dev server (port 5173) and a generic React dev server
# (port 3000) to make cross-origin requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
# Evaluation & Retrieval endpoints are served under /api/v1 for standard API versioning,
# and also mounted at root so calls to /evaluations/ and /health work directly.
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="", include_in_schema=False)
app.include_router(retrieval_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check", tags=["Health"])
def root_health():
    """Liveness probe at /health."""
    return {"status": "healthy", "service": "Verity API"}


@app.get("/", include_in_schema=False)
def root():
    """Redirect browsers hitting the root to the interactive API docs."""
    return {"message": "Welcome to Verity API. Visit /docs for the interactive API documentation."}
