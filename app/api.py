from contextlib import asynccontextmanager
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.api_schemas import ChatStreamRequest, HealthResponse
from app.cli import ChatbotApp


def _to_sse(data: str, event: str | None = None) -> str:
    """Format text into an SSE-compliant payload."""
    lines = str(data).splitlines() or [""]
    event_line = f"event: {event}\n" if event else ""
    data_lines = "".join(f"data: {line}\n" for line in lines)
    return f"{event_line}{data_lines}\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize once so model/db clients are reused across requests.
    app.state.chatbot = ChatbotApp()
    yield


app = FastAPI(title="Medical RAG Streaming API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    chatbot: ChatbotApp = app.state.chatbot

    async def event_generator():
        try:
            async for event, payload in chatbot.stream_answer_events(request.query, request.role):
                if not payload:
                    continue
                # Wrap payload into structured JSON so frontend receives
                #: {"type": "text|trace|...", "text": "..."}
                if event == "trace":
                    obj = {"type": "trace", "text": payload}
                elif event == "chunk":
                    obj = {"type": "text", "text": payload}
                else:
                    obj = {"type": event or "text", "text": payload}

                yield _to_sse(json.dumps(obj, ensure_ascii=False))
            yield _to_sse(json.dumps({"type": "done"}, ensure_ascii=False))
        except Exception as exc:
            yield _to_sse(json.dumps({"type": "error", "text": str(exc)}, ensure_ascii=False))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
