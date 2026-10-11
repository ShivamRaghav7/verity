"""Local FastAPI application and explicit administrator boundary."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from time import perf_counter

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import AskRequest, CorrectionCreate
from .service import StructuredAssistant
from .store import DomainError, Store

STATIC = Path(__file__).parent / "static"


def create_app(db_path=None, admin_token=None, comparison_runner=None):
    token = admin_token or os.environ.get("EDITGUARD_ADMIN_TOKEN", "local-demo-only")
    path = Path(db_path or os.environ.get("EDITGUARD_DB", "runtime/editguard.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    store.seed()
    assistant = StructuredAssistant(store)
    neural_lock = RLock()
    runner = comparison_runner

    def neural_runner():
        nonlocal runner
        with neural_lock:
            if runner is None:
                config = os.environ.get("EDITGUARD_INFERENCE_CONFIG")
                if not config:
                    raise HTTPException(503, "Neural inference is not enabled. Set EDITGUARD_INFERENCE_CONFIG in the isolated ML environment.")
                from .comparison import ComparisonRunner
                try:
                    runner = ComparisonRunner.from_config(config)
                except Exception as exc:
                    raise HTTPException(503, f"Neural configuration could not load: {exc}") from exc
            return runner
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="EditGuard", version="0.2.0", lifespan=lifespan,
                  description="Controlled support knowledge editing; structured workflow and optional frozen neural comparisons.")
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
                "models": {"scope": "trained" if runner and runner.scope else "not_trained",
                           "answer": "trained" if runner and runner.answer_model else "not_trained"},
                "neural_inference": "loaded" if runner else "configured_not_loaded" if os.environ.get("EDITGUARD_INFERENCE_CONFIG") else "disabled",
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
        if request.mode == "structured":
            return assistant.ask(request)
        started = perf_counter()
        comparison = neural_runner()
        with store.snapshot():
            facts, edits, conflicts, revision = store.facts(), store.edits(), store.conflicts(), store.revision()
        configuration = "trained_scope_editor" if request.mode == "learned" else request.mode
        context = request.model_dump(mode="json", exclude={"question", "mode"}, exclude_none=True)
        result = comparison.answer(request.question, context, facts, edits, configuration=configuration,
                                   revision=revision, conflicts=conflicts)
        if result["status"] == "unavailable":
            raise HTTPException(503, result["error"])
        before = result if configuration == "unchanged_base" else comparison.answer(
            request.question, context, facts, edits, configuration="unchanged_base", revision=revision)
        result.update(before_answer=before["answer"], before_answer_kind=before["answer_kind"],
                      before_trace=before, fact=None,
                      applicable_corrections=[edit for edit in edits if edit["edit_id"] == result["selected_edit_id"]],
                      explanation="Model-generated answer. Supplied sources and corrections do not verify generated prose.",
                      api_latency_ms=round((perf_counter() - started) * 1000, 3))
        # Copy before_trace when it is the result itself, avoiding a JSON cycle.
        if before is result:
            result["before_trace"] = None
        return result

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
