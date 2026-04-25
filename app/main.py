from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.container import AppContainer


def create_app(*, container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title="RAG Template", version="0.1.0")
    app.state.container = container or AppContainer()
    app.include_router(router)
    return app


app = create_app()
