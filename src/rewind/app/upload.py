import html
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile
from fastapi.responses import HTMLResponse

from rewind.core.conversation import Conversation

MAX_UPLOAD_BYTES = 2 * 1024**3

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rewind upload</title>
<style>body{{font:16px system-ui,sans-serif;max-width:32rem;margin:3rem auto;padding:0 1rem}}
button{{font:inherit;padding:.5rem 1rem}}</style></head><body>{body}</body></html>"""


def _page(body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body), status_code=status)


_EXPIRED = "<h1>This link has expired</h1><p>Ask Rewind in your chat for a new one.</p>"


# The page the signed upload links point to; T6 replaces the bare form later.
def upload_router(conversation: Conversation) -> APIRouter:
    router = APIRouter()

    @router.get("/u/{token}")
    async def form(token: str) -> HTMLResponse:
        line = await conversation.upload_target(token)
        if line is None:
            return _page(_EXPIRED, 404)
        return _page(
            f"<h1>Upload footage</h1><p>For <b>{html.escape(line)}</b>. One file, up to 2 GB.</p>"
            '<form method="post" enctype="multipart/form-data">'
            '<p><input type="file" name="file" accept="video/*" required></p>'
            "<p><button>Upload</button></p></form>"
        )

    @router.post("/u/{token}")
    async def upload(token: str, file: UploadFile) -> HTMLResponse:
        if await conversation.upload_target(token) is None:
            return _page(_EXPIRED, 404)
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            return _page("<h1>That file is over 2 GB</h1>", 413)
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / "upload"
            with local.open("wb") as destination:
                shutil.copyfileobj(file.file, destination)
            accepted = await conversation.receive_upload(
                token, file.filename or "upload.mp4", local
            )
        if not accepted:
            return _page(_EXPIRED, 404)
        return _page("<h1>Upload finished</h1><p>Rewind will message you in your chat.</p>")

    return router
