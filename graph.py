import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from tools import format_context, search_local_documents, thread_has_documents, list_thread_documents, web_search


def overwrite(_: Any, new: Any) -> Any:
    return new


class SentinelState(TypedDict, total=False):
    original_query: str
    thread_id: str
    chat_history: Annotated[List[str], operator.add]
    document_ids: List[str]
    current_sub_question: str
    retrieved_context: str
    current_draft: str
    critic_feedback: str
    critic_status: str
    critic_grounded: bool
    critic_citations_valid: bool
    critic_completeness: str
    sources: List[Dict[str, Any]]
    loop_count: int
    human_feedback: str
    review_reason: str
    retrieval_mode: str
    retrieval_grade: str



class GraderOutput(BaseModel):
    relevance: str = Field(description="RELEVANT or IRRELEVANT")
    reasoning: str = Field(description="Why the retrieved evidence is or is not relevant")


class CriticOutput(BaseModel):
    status: str = Field(description="APPROVED or REJECTED")
    grounded: bool = Field(description="Whether factual claims are supported by the supplied context")
    citations_valid: bool = Field(description="Whether cited source IDs are present in the context")
    completeness: str = Field(description="HIGH, MEDIUM, or LOW")
    feedback: str = Field(description="Specific correction instructions")


class RouteOutput(BaseModel):
    intent: str = Field(description="CHITCHAT or RESEARCH")


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
structured_grader = llm.with_structured_output(GraderOutput)
structured_critic = llm.with_structured_output(CriticOutput)
structured_router = llm.with_structured_output(RouteOutput)


async def planner_node(state: Dict[str, Any]):
    history = "\n".join(state.get("chat_history", [])[-12:-1])
    human_feedback = state.get("human_feedback", "").strip()
    critic_feedback = state.get("critic_feedback", "").strip()

    if not history.strip() and not human_feedback and not critic_feedback:
        return {"current_sub_question": state["original_query"]}

    prompt = f"""
You are the query planner for Sentinel, an AI research assistant.
Rewrite the user's latest request into one precise, standalone research query.
Preserve the user's actual goal. Do not answer the question.

Conversation history:
{history or '[none]'}

Latest user request:
{state['original_query']}

Human steering feedback:
{human_feedback or '[none]'}

Previous critic feedback:
{critic_feedback or '[none]'}

Return ONLY the standalone search query.
"""
    response = await llm.ainvoke(prompt)
    return {"current_sub_question": response.content.strip()}


async def retriever_node(state: Dict[str, Any]):
    query = state["current_sub_question"]
    feedback_parts = [state.get("critic_feedback", "").strip(), state.get("human_feedback", "").strip()]
    guidance = " ".join(part for part in feedback_parts if part)
    retrieval_query = f"{query}. Correction guidance: {guidance}" if guidance else query

    local_results = search_local_documents(
        state["thread_id"],
        retrieval_query,
        document_ids=state.get("document_ids") or None,
    )
    has_uploaded_docs = thread_has_documents(state["thread_id"])

    if local_results:
        return {
            "retrieved_context": format_context(local_results),
            "sources": [result["source"] for result in local_results],
            "retrieval_mode": "LOCAL_AND_UPLOAD" if has_uploaded_docs else "LOCAL",
        }

    # Only use the web when the local/uploaded corpus truly has no evidence.
    web_results = await web_search(retrieval_query)
    return {
        "retrieved_context": format_context(web_results),
        "sources": [result["source"] for result in web_results],
        "retrieval_mode": "WEB_FALLBACK",
    }


async def crag_grader_node(state: Dict[str, Any]):
    context = state.get("retrieved_context", "")
    if not context or context.startswith("No relevant"):
        web_results = await web_search(state["current_sub_question"])
        return {
            "retrieved_context": format_context(web_results),
            "sources": [result["source"] for result in web_results],
            "retrieval_mode": "WEB_FALLBACK",
        }

    evaluation = await structured_grader.ainvoke(
        f"""
Question: {state['current_sub_question']}
Retrieved evidence:
{context}

Determine whether the evidence is relevant enough to answer the question.
Be conservative: irrelevant or weakly related evidence should be marked IRRELEVANT.
"""
    )

    if evaluation.relevance.upper() == "IRRELEVANT":
        # With uploaded documents, preserve document grounding. The generator should
        # explicitly say the document does not contain enough information rather than
        # silently replacing the user's document with unrelated web content.
        if thread_has_documents(state["thread_id"]):
            return {"retrieval_grade": evaluation.reasoning, "retrieval_mode": "DOCUMENT_INSUFFICIENT"}

        web_results = await web_search(state["current_sub_question"])
        return {
            "retrieved_context": format_context(web_results),
            "sources": [result["source"] for result in web_results],
            "retrieval_mode": "WEB_FALLBACK",
        }

    return {"retrieval_grade": evaluation.reasoning}


async def generator_node(state: Dict[str, Any]):
    previous_feedback = state.get("critic_feedback", "").strip()
    feedback_block = previous_feedback if previous_feedback else "No previous critic feedback."

    prompt = f"""
You are Sentinel, an evidence-grounded AI research assistant.

Question:
{state['current_sub_question']}

Evidence:
{state['retrieved_context']}

Previous critic feedback:
{feedback_block}

Write the best possible answer using ONLY the supplied evidence.

Rules:
1. Do not invent facts, statistics, quotations, or sources.
2. Every substantive factual claim must be supported by at least one supplied source.
3. Cite claims using the exact source IDs such as [S1], [W2].
4. If the evidence is insufficient, explicitly say what cannot be established.
5. Separate evidence from inference when an inference is unavoidable.
6. Do not cite a source ID that does not exist in the evidence.
7. End with a "Sources" section listing only the sources actually cited.

Return only the final answer.
"""
    response = await llm.ainvoke(prompt)
    return {"current_draft": response.content.strip()}


async def critic_node(state: Dict[str, Any]):
    evaluation = await structured_critic.ainvoke(
        f"""
You are Sentinel's final quality critic.

Question:
{state['current_sub_question']}

Draft:
{state['current_draft']}

Available evidence:
{state['retrieved_context']}

Check:
- Groundedness: every substantive claim is supported by evidence.
- Citation validity: every citation ID exists in the evidence.
- Completeness: the answer addresses the question adequately given the evidence.
- Hallucination: no unsupported details are introduced.

Reject when material claims are unsupported, citations are fabricated/invalid, or the response misses the core request.
Give concrete correction instructions.
"""
    )
    new_loop_count = int(state.get("loop_count", 0)) + 1
    return {
        "critic_status": evaluation.status.upper(),
        "critic_feedback": evaluation.feedback.strip(),
        "critic_grounded": evaluation.grounded,
        "critic_citations_valid": evaluation.citations_valid,
        "critic_completeness": evaluation.completeness.upper(),
        "loop_count": new_loop_count,
        "review_reason": (
            "critic_approved"
            if evaluation.status.upper() == "APPROVED"
            else ("max_retries" if new_loop_count >= 3 else "critic_rejected")
        ),
    }


async def human_review_node(state: Dict[str, Any]):
    return {}


async def synthesizer_node(state: Dict[str, Any]):
    final_report = state["current_draft"]
    return {
        "chat_history": [f"Sentinel: {final_report}"],
        "human_feedback": "",
        "critic_feedback": "",
    }


async def document_status_node(state: Dict[str, Any]):
    docs = list_thread_documents(state["thread_id"])
    if not docs:
        return {
            "current_draft": "I don't have any indexed documents for this chat. Please upload a PDF or TXT file first."
        }

    lines = ["Yes. Sentinel can access the document(s) you uploaded to this chat:"]
    for doc in docs:
        lines.append(f"- **{doc['filename']}** — {doc['chunks']} text chunks indexed")
    lines.append("\nAsk me to summarize it, explain a section, extract information, or answer questions from it.")
    return {"current_draft": "\n".join(lines), "chat_history": [f"Sentinel: {'\n'.join(lines)}"]}


async def chitchat_node(state: Dict[str, Any]):
    history = "\n".join(state.get("chat_history", [])[-12:])
    response = await llm.ainvoke(
        f"""
You are Sentinel, a helpful AI assistant.
Use the recent conversation to maintain context and answer naturally.
Do not invent personal facts about the user.

Conversation:
{history}

Latest message:
{state['original_query']}

Respond naturally and concisely.
"""
    )
    return {
        "current_draft": response.content.strip(),
        "chat_history": [f"Sentinel: {response.content.strip()}"],
    }


async def route_initial_query(state: Dict[str, Any]):
    history = "\n".join(state.get("chat_history", [])[-12:])
    has_document = bool(state.get("document_ids")) or thread_has_documents(state["thread_id"])

    # Deterministic handling for document-presence questions. These should never
    # fall through to open-web retrieval, because the correct answer is based on
    # the application's indexed-document state.
    q = state["original_query"].lower().strip()
    document_presence_phrases = (
        "can you see the document",
        "can you see my document",
        "can you access the document",
        "can you access my document",
        "can you read the document",
        "can you read my document",
        "do you see the document",
        "is the document uploaded",
        "did you get the document",
        "can you see the file",
        "can you access the file",
        "can you read the file",
        "is the file uploaded",
    )
    if has_document and any(phrase in q for phrase in document_presence_phrases):
        return "document_status"

    prompt = f"""
Classify the user's latest message as CHITCHAT or RESEARCH.

CHITCHAT: greetings, casual conversation, thanks, goodbyes, or non-factual personal conversation.
RESEARCH: factual questions, explanations, summaries, analysis, calculations, code questions, document questions, or anything requiring evidence.
If an uploaded document is attached, choose RESEARCH.

Conversation:
{history}

Latest message:
{state['original_query']}
Uploaded document attached: {'YES' if has_document else 'NO'}
"""
    try:
        result = await structured_router.ainvoke(prompt)
        return "chitchat" if result.intent.upper() == "CHITCHAT" and not has_document else "planner"
    except Exception:
        return "planner"


def route_critic(state: Dict[str, Any]):
    if state.get("critic_status") == "APPROVED":
        return "review"
    if int(state.get("loop_count", 0)) >= 3:
        return "review"
    return "retry"


builder = StateGraph(SentinelState)

# Nodes
builder.add_node("planner", planner_node)
builder.add_node("retriever", retriever_node)
builder.add_node("crag_grader", crag_grader_node)
builder.add_node("generator", generator_node)
builder.add_node("critic", critic_node)
builder.add_node("human_review", human_review_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("document_status", document_status_node)

# Initial routing
builder.add_conditional_edges(
    START,
    route_initial_query,
    {"chitchat": "chitchat", "planner": "planner", "document_status": "document_status"},
)

# Research path
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "crag_grader")
builder.add_edge("crag_grader", "generator")
builder.add_edge("generator", "critic")

# Self-RAG feedback loop + HITL
builder.add_conditional_edges(
    "critic",
    route_critic,
    {"retry": "planner", "review": "human_review"},
)

builder.add_edge("human_review", "synthesizer")
builder.add_edge("synthesizer", END)
builder.add_edge("chitchat", END)
builder.add_edge("document_status", END)
