import hashlib
import json
import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHATS_FILE = "chats.json"

st.set_page_config(page_title="Sentinel Research Pipeline", page_icon="🛡️", layout="wide")
st.title("🛡️ Sentinel Research Pipeline")


def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_chats(chats_dict):
    tmp = CHATS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chats_dict, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHATS_FILE)


def backend_get_history(thread_id):
    response = requests.get(f"{BACKEND_URL}/history/{thread_id}", timeout=20)
    response.raise_for_status()
    return response.json().get("chat_history", [])


def parse_history(raw_history):
    parsed = []
    for msg in raw_history:
        if msg.startswith("User: "):
            parsed.append({"role": "user", "content": msg[6:]})
        elif msg.startswith("Sentinel: "):
            parsed.append({"role": "assistant", "content": msg[10:]})
    return parsed


all_chats = load_chats()

if "thread_id" not in st.session_state:
    thread_from_url = st.query_params.get("thread_id")
    st.session_state.thread_id = thread_from_url or str(uuid.uuid4())
    st.query_params["thread_id"] = st.session_state.thread_id

thread_id = st.session_state.thread_id

if "chat_history" not in st.session_state or st.session_state.get("last_loaded_thread") != thread_id:
    try:
        st.session_state.chat_history = parse_history(backend_get_history(thread_id))
    except requests.RequestException:
        st.session_state.chat_history = []
    st.session_state.last_loaded_thread = thread_id
    st.session_state.agent_status = "idle"
    st.session_state.draft_text = ""
    st.session_state.document_ids = []
    st.session_state.upload_hash = None
    st.session_state.review_reason = ""
    try:
        docs_response = requests.get(f"{BACKEND_URL}/documents/{thread_id}", timeout=20)
        docs_response.raise_for_status()
        st.session_state.document_ids = [d["document_id"] for d in docs_response.json().get("documents", [])]
    except requests.RequestException:
        pass

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_thread = str(uuid.uuid4())
        st.session_state.thread_id = new_thread
        st.query_params["thread_id"] = new_thread
        st.session_state.chat_history = []
        st.session_state.agent_status = "idle"
        st.session_state.draft_text = ""
        st.session_state.document_ids = []
        st.session_state.upload_hash = None
        st.rerun()

    st.subheader("Past Chats")
    for tid, title in reversed(list(all_chats.items())):
        if st.button(title, key=f"chat_{tid}", use_container_width=True):
            st.session_state.thread_id = tid
            st.query_params["thread_id"] = tid
            st.rerun()

    st.divider()
    st.header("📁 Document Ingestion")
    existing_docs = st.session_state.get("document_ids", [])
    if existing_docs:
        st.caption(f"{len(existing_docs)} document(s) indexed for this chat.")
        try:
            doc_response = requests.get(f"{BACKEND_URL}/documents/{thread_id}", timeout=10)
            if doc_response.ok:
                for doc in doc_response.json().get("documents", []):
                    st.caption(f"📄 {doc['filename']} — {doc['chunks']} chunks")
        except requests.RequestException:
            pass
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        st.success(f"Loaded: {uploaded_file.name}")

        if st.session_state.get("upload_hash") != file_hash:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ingest",
                    params={"thread_id": thread_id},
                    files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.upload_hash = file_hash
                doc_id = result["document_id"]
                if doc_id not in st.session_state.document_ids:
                    st.session_state.document_ids.append(doc_id)
                st.caption(f"Indexed {result['chunks']} chunks.")
            except requests.RequestException as exc:
                st.error(f"Document indexing failed: {exc}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.agent_status == "awaiting_approval":
    st.markdown("---")
    st.subheader("🤖 Sentinel Draft Review")
    if st.session_state.get("review_reason") == "max_retries":
        st.warning("Sentinel could not fully verify this answer after several automatic corrections. Please review it carefully.")
    else:
        st.info("Sentinel paused for human review before committing the answer to conversation memory.")

    edited_draft = st.text_area(
        "Review/Edit generated draft:",
        value=st.session_state.draft_text,
        height=320,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve & Finalize", use_container_width=True, type="primary"):
            with st.spinner("Finalizing..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/resume",
                        json={"thread_id": thread_id, "action": "approve", "edited_draft": edited_draft},
                        timeout=120,
                    )
                    response.raise_for_status()
                    final_report = response.json().get("final_report", edited_draft)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_report})
                    st.session_state.draft_text = ""
                    st.session_state.agent_status = "idle"
                    st.session_state.review_reason = ""
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Finalization failed: {exc}")

    with col2:
        with st.popover("❌ Reject & Redirect Agent", use_container_width=True):
            human_feedback = st.text_area(
                "What should Sentinel change?",
                placeholder="e.g. Remove the unsupported claim about X and focus on the evidence in source [S2].",
                key="human_feedback_box",
            )
            if st.button("Force Recalculation", type="primary"):
                if not human_feedback.strip():
                    st.warning("Please provide steering feedback.")
                else:
                    with st.spinner("Rerouting Agent..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/resume",
                                json={
                                    "thread_id": thread_id,
                                    "action": "reject",
                                    "edited_draft": human_feedback,
                                },
                                timeout=120,
                            )
                            response.raise_for_status()
                            new_draft = response.json().get("final_report", "")
                            st.session_state.draft_text = new_draft
                            st.session_state.agent_status = "awaiting_approval"
                            st.session_state.review_reason = "critic_rejected"
                            st.rerun()
                        except requests.RequestException as exc:
                            st.error(f"Rerouting failed: {exc}")

if user_query := st.chat_input(
    "Ask Sentinel anything...",
    disabled=(st.session_state.agent_status != "idle"),
):
    if thread_id not in all_chats:
        all_chats[thread_id] = user_query[:40] + ("..." if len(user_query) > 40 else "")
        save_chats(all_chats)

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.agent_status = "running"
    st.session_state.draft_text = ""

    with st.status("Executing Sentinel...", expanded=True) as status_box:
        try:
            response = requests.post(
                f"{BACKEND_URL}/stream",
                json={
                    "query": user_query,
                    "thread_id": thread_id,
                    "document_ids": st.session_state.document_ids,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            interrupted = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    update = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "error" in update:
                    st.error(f"Backend error: {update['error']}")
                    st.session_state.agent_status = "idle"
                    break

                if "__interrupt__" in update:
                    interrupted = True
                    status_box.update(label="Awaiting Human Review", state="running")
                    break

                for node_name, state_data in update.items():
                    st.write(f"✅ **{node_name.upper()}** completed.")
                    if isinstance(state_data, dict):
                        if state_data.get("current_draft"):
                            st.session_state.draft_text = state_data["current_draft"]
                        if state_data.get("review_reason"):
                            st.session_state.review_reason = state_data["review_reason"]

            if interrupted:
                st.session_state.agent_status = "awaiting_approval"
                st.rerun()
            else:
                if st.session_state.draft_text:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": st.session_state.draft_text}
                    )
                st.session_state.agent_status = "idle"
                st.rerun()

        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")
            st.session_state.agent_status = "idle"
