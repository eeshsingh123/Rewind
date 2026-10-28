from fastapi import APIRouter, FastAPI

from rewind.app.settings import Settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Rewind")
    app.state.settings = settings or Settings()
    app.include_router(router)
    return app
