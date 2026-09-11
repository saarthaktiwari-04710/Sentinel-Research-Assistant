import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from graph import builder
from tools import ingest_document, list_thread_documents

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = BASE_DIR / "checkpoints.sqlite"
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        await checkpointer.setup()
        agent = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review"],
        )
        yield


app = FastAPI(title="Sentinel Research Pipeline", version="2.0.0", lifespan=lifespan)


class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str = Field(min_length=1, max_length=200)
    document_ids: list[str] = []


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    action: str
    edited_draft: str = ""


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    state = await agent.aget_state(_config(thread_id))
    if state and getattr(state, "values", None):
        return {"chat_history": state.values.get("chat_history", [])}
    return {"chat_history": []}


@app.get("/documents/{thread_id}")
async def get_documents(thread_id: str):
    return {"documents": list_thread_documents(thread_id)}


@app.post("/ingest")
async def ingest(thread_id: str, file: UploadFile = File(...)):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    filename = file.filename or "uploaded_document"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    data = await file.read()
    max_bytes = 15 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Document is larger than the 15 MB limit")

    try:
        result = ingest_document(thread_id, filename, data)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc


@app.post("/stream")
async def stream_agent(request: RunRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")

    async def sse_generator():
        config = _config(request.thread_id)
        inputs = {
            "original_query": request.query,
            "thread_id": request.thread_id,
            "chat_history": [f"User: {request.query}"],
            "document_ids": request.document_ids,
            "current_sub_question": "",
            "retrieved_context": "",
            "current_draft": "",
            "critic_feedback": "",
            "critic_status": "",
            "critic_grounded": False,
            "critic_citations_valid": False,
            "critic_completeness": "",
            "sources": [],
            "loop_count": 0,
            "human_feedback": "",
            "review_reason": "",
            "retrieval_mode": "",
        }
        try:
            async for chunk in agent.astream(inputs, config, stream_mode="updates"):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/resume")
async def resume_agent(request: ResumeRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")

    config = _config(request.thread_id)
    current = await agent.aget_state(config)
    if not current or not getattr(current, "values", None):
        raise HTTPException(status_code=404, detail="No paused run exists for this thread")

    action = request.action.lower().strip()
    if action == "approve":
        draft = request.edited_draft.strip()
        if not draft:
            raise HTTPException(status_code=400, detail="edited_draft cannot be empty")
        await agent.aupdate_state(config, {"current_draft": draft})
        try:
            async for _ in agent.astream(None, config, stream_mode="updates"):
                pass
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc

    elif action == "reject":
        feedback = request.edited_draft.strip()
        if not feedback:
            raise HTTPException(status_code=400, detail="Please provide steering feedback")
        await agent.aupdate_state(
            config,
            {
                "human_feedback": feedback,
                "critic_feedback": feedback,
                "critic_status": "REJECTED",
                "loop_count": 0,
                "review_reason": "human_rejection",
            },
            as_node="critic",
        )
        try:
            async for _ in agent.astream(None, config, stream_mode="updates"):
                pass
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Rerouting failed: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    final_state = await agent.aget_state(config)
    return {
        "final_report": final_state.values.get("current_draft", "") if final_state else ""
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)