from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.logging import logger
from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.api.routes import feedback

# 👇 ADD THIS
# from app.services.auto_ingest import run_ingestion_if_needed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -----------------------------
    # STARTUP
    # -----------------------------
    logger.info("Startup complete (ingestion disabled in production)")
    yield

    # -----------------------------
    # SHUTDOWN
    # -----------------------------
    logger.info("🛑 Shutting down Enterprise RAG Copilot API")


app = FastAPI(
    title="Enterprise RAG Copilot",
    version="0.1.0",
    lifespan=lifespan,
)

# -----------------------------
# ROUTES
# -----------------------------
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(feedback.router)


@app.get("/")
def root():
    return {"message": "RAG Copilot is running"}
