"""Local FastAPI application and explicit administrator boundary."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import AskRequest, CorrectionCreate
from .service import StructuredAssistant
from .store import DomainError, Store

STATIC = Path(__file__).parent / "static"


def create_app(db_path=None, admin_token=None):
    token = admin_token or os.environ.get("EDITGUARD_ADMIN_TOKEN", "local-demo-only")
    path = Path(db_path or os.environ.get("EDITGUARD_DB", "runtime/editguard.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    store.seed()
    assistant = StructuredAssistant(store)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="EditGuard", version="0.1.0", lifespan=lifespan,
                  description="Controlled support knowledge editing; structured vertical slice.")
    app.state.store = store

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        return JSONResponse(status_code=exc.code, content={"detail": str(exc)})

    def administrator(x_admin_token: str = Header(default="")):
        if not secrets.compare_digest(x_admin_token, token):
            raise HTTPException(401, "Administrator token required.")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "revision": store.revision(),
                "models": {"scope": "not_trained", "answer": "not_trained"},
                "answer_mode": "deterministic_structured"}

    @app.get("/api/facts")
    def facts():
        with store.snapshot():
            return {"facts": store.facts(), "revision": store.revision()}

    @app.get("/api/edits")
    def edits():
        with store.snapshot():
            return {"edits": store.edits(), "events": store.events(), "revision": store.revision()}

    @app.get("/api/conflicts")
    def conflicts():
        with store.snapshot():
            return {"conflicts": store.conflicts(), "revision": store.revision()}

    @app.post("/api/edits", status_code=201, dependencies=[Depends(administrator)])
    def create_edit(correction: CorrectionCreate):
        return store.create_correction(correction)

    @app.post("/api/edits/{edit_id}/approve", dependencies=[Depends(administrator)])
    def approve(edit_id: str):
        return store.approve(edit_id)

    @app.post("/api/edits/{edit_id}/withdraw", dependencies=[Depends(administrator)])
    def withdraw(edit_id: str):
        return store.withdraw(edit_id)

    @app.post("/api/ask")
    def ask(request: AskRequest):
        if request.mode != "structured":
            raise HTTPException(503, "Learned mode requires trained, calibrated scope and answer checkpoints; neither is installed.")
        return assistant.ask(request)

    @app.get("/api/evaluation")
    def evaluation():
        from .evaluation import run_slice_regression
        return {**run_slice_regression(), "revision": store.revision()}

    @app.get("/api/export")
    def export():
        with store.snapshot():
            payload = {"schema_version": 1, "dataset": "controlled_fictional_demonstration",
                       "models": health()["models"], "facts": store.facts(), "edits": store.edits(),
                       "events": store.events(), "revision": store.revision()}
        payload["evaluation"] = evaluation()
        return payload

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return app
