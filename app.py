import io
import json
import os
import time
import requests
import streamlit as st
import pymupdf
from pypdf import PdfReader

# Page Configuration
st.set_page_config(
    page_title="PDF LLM Parser & Token Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .context-bar-container {
        width: 100%;
        background-color: rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        overflow: hidden;
        height: 22px;
        display: flex;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .bar-cached {
        background-color: #22c55e;
        height: 100%;
        transition: width 0.3s ease;
    }
    .bar-uncached {
        background-color: #3b82f6;
        height: 100%;
        transition: width 0.3s ease;
    }
    .bar-output {
        background-color: #a855f7;
        height: 100%;
        transition: width 0.3s ease;
    }
    .bar-remaining {
        background-color: transparent;
        height: 100%;
    }
    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .pdf-meta-box {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 0.9rem 1.1rem;
        border-radius: 0.6rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
def fix_pua(text: str) -> str:
    """Decodes characters shifted into Unicode Private Use Area (0xF000 - 0xF0FF)."""
    return "".join(
        chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c
        for c in text
    )

def extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    """Extracts and normalizes text from PDF bytes using pymupdf with pypdf fallback."""
    text_parts = []
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned_text = fix_pua(raw_text)
            text_parts.append(f"--- Page {page_num} ---\n{cleaned_text}")
    except Exception:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = fix_pua(raw_text)
            text_parts.append(f"--- Page {page_num} ---\n{cleaned_text}")
    return "\n\n".join(text_parts)

def resolve_ollama_host(base_url: str) -> str:
    """Normalize base URL so that it points to the Ollama root host (without /v1)."""
    clean_url = base_url.rstrip("/")
    if clean_url.endswith("/v1"):
        clean_url = clean_url[:-3]
    return clean_url

def query_ollama_models(ollama_host: str):
    """Fetch installed models from Ollama."""
    try:
        res = requests.get(f"{ollama_host}/api/tags", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ LLM & Context Setup")

    default_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_base_url = st.text_input(
        "Ollama Endpoint",
        value=default_base_url,
        help="OpenAI-compatible or native Ollama URL (e.g. http://localhost:11434 or http://ollama:11434)",
    )
    ollama_host = resolve_ollama_host(ollama_base_url)

    available_models = query_ollama_models(ollama_host)
    default_model = os.getenv("OLLAMA_MODEL", "medgemma1.5")
    
    if available_models:
        default_index = available_models.index(default_model) if default_model in available_models else 0
        selected_model = st.selectbox("Model", available_models, index=default_index)
    else:
        selected_model = st.text_input("Model Name", value=default_model)

    # Context window limit setting
    st.markdown("### Context Window Limit")
    num_ctx = st.select_slider(
        "Context Window (num_ctx)",
        options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
        value=8192,
        help="Size of the KV context window allocated in Ollama memory",
    )

    st.markdown("---")
    st.subheader("Prompt Persona")
    parser_mode = st.selectbox(
        "System Prompt Persona",
        [
            "Medical Document Parser (Strict)",
            "General Document Assistant",
            "Structured JSON Extractor",
        ],
        index=0,
    )

    system_prompts = {
        "Medical Document Parser (Strict)": (
            "You are a medical document parser. Extract only data explicitly stated in the PDF text — never infer, "
            "diagnose, guess, or fill in typical values; missing fields must be null. "
            "Preserve numbers, units, dates, and medication names/dosages exactly as written, "
            "with no reformatting or normalization. If text is unclear or garbled (e.g. from OCR), "
            "mark it `'unclear': true` instead of guessing. "
            "Ignore headers, footers, and boilerplate unless clinically relevant."
        ),
        "General Document Assistant": (
            "You are a helpful assistant. Answer questions strictly using the information "
            "contained in the provided document. If the answer cannot be found in the document, say so clearly."
        ),
        "Structured JSON Extractor": (
            "You are an expert information extractor. Extract all relevant findings, values, and entities "
            "from the document into valid JSON format without markdown ticks. Rely only on explicitly stated facts."
        ),
    }
    selected_system_prompt = system_prompts[parser_mode]

    if st.button("🧹 Clear Chat & Token Metrics", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_token_metrics = None
        st.rerun()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if "last_token_metrics" not in st.session_state:
    st.session_state.last_token_metrics = None

# Header Section
st.markdown("## 🔬 PDF Document Parser & Context Window Inspector")
st.caption("Upload documents, parse medical records or papers, and track exact Ollama prompt evaluation, KV cache hits, and generation speeds.")

# Context Window & Token Visualization Banner
with st.container():
    metrics = st.session_state.last_token_metrics

    # Fallbacks or actual metrics
    cached_tokens = metrics.get("prompt_eval_cached_count", 0) if metrics else 0
    total_prompt_tokens = metrics.get("prompt_eval_count", 0) if metrics else 0
    # In Ollama: prompt_eval_count is either total prompt tokens or evaluated ones depending on version
    # Calculate uncached portion
    uncached_tokens = max(0, total_prompt_tokens - cached_tokens) if cached_tokens > 0 else total_prompt_tokens
    output_tokens = metrics.get("eval_count", 0) if metrics else 0
    total_used_tokens = cached_tokens + uncached_tokens + output_tokens

    # Durations
    prompt_eval_dur_s = (metrics.get("prompt_eval_duration", 0) / 1e9) if metrics else 0.0
    eval_dur_s = (metrics.get("eval_duration", 0) / 1e9) if metrics else 0.0
    total_dur_s = (metrics.get("total_duration", 0) / 1e9) if metrics else 0.0

    eval_speed = (output_tokens / eval_dur_s) if eval_dur_s > 0 else 0.0
    cache_hit_pct = ((cached_tokens / (cached_tokens + uncached_tokens)) * 100) if (cached_tokens + uncached_tokens) > 0 else 0.0

    st.markdown("### 📊 Context Window & Token Allocation")

    # Metric Cards Row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🟢 Cached Input</div>
                <div class="metric-value" style="color: #22c55e;">{cached_tokens:,}</div>
                <small>{cache_hit_pct:.1f}% hit</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🔵 Uncached Input</div>
                <div class="metric-value" style="color: #3b82f6;">{uncached_tokens:,}</div>
                <small>{prompt_eval_dur_s:.2f}s prefill</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🟣 Output (Eval)</div>
                <div class="metric-value" style="color: #a855f7;">{output_tokens:,}</div>
                <small>{eval_speed:.1f} t/s</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📦 Total Tokens</div>
                <div class="metric-value">{total_used_tokens:,}</div>
                <small>of {num_ctx:,} max</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col5:
        context_usage_pct = (total_used_tokens / num_ctx) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📈 Window Fill</div>
                <div class="metric-value" style="color: {'#ef4444' if context_usage_pct > 80 else '#eab308' if context_usage_pct > 50 else '#22c55e'};">{context_usage_pct:.1f}%</div>
                <small>{max(0, num_ctx - total_used_tokens):,} free</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">⏱️ Total Latency</div>
                <div class="metric-value">{total_dur_s:.2f}s</div>
                <small>{'Idle' if not metrics else 'Last request'}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Multi-Segment Context Bar
    pct_cached = min(100.0, (cached_tokens / num_ctx) * 100)
    pct_uncached = min(100.0 - pct_cached, (uncached_tokens / num_ctx) * 100)
    pct_output = min(100.0 - pct_cached - pct_uncached, (output_tokens / num_ctx) * 100)
    pct_remaining = max(0.0, 100.0 - (pct_cached + pct_uncached + pct_output))

    st.markdown(
        f"""
        <div class="context-bar-container">
            <div class="bar-cached" style="width: {pct_cached}%;" title="Cached Input: {cached_tokens} tokens ({pct_cached:.1f}%)"></div>
            <div class="bar-uncached" style="width: {pct_uncached}%;" title="Uncached Input: {uncached_tokens} tokens ({pct_uncached:.1f}%)"></div>
            <div class="bar-output" style="width: {pct_output}%;" title="Output Generation: {output_tokens} tokens ({pct_output:.1f}%)"></div>
            <div class="bar-remaining" style="width: {pct_remaining}%;"></div>
        </div>
        <div style="font-size: 0.82rem; color: #888; display: flex; justify-content: space-between; margin-bottom: 1.5rem;">
            <div>
                <span class="legend-dot" style="background:#22c55e;"></span>Cached ({pct_cached:.1f}%) &nbsp;&nbsp;
                <span class="legend-dot" style="background:#3b82f6;"></span>Uncached Input ({pct_uncached:.1f}%) &nbsp;&nbsp;
                <span class="legend-dot" style="background:#a855f7;"></span>Output ({pct_output:.1f}%) &nbsp;&nbsp;
                <span class="legend-dot" style="background:rgba(128,128,128,0.2);"></span>Available Free Space ({pct_remaining:.1f}%)
            </div>
            <div><b>Capacity: {num_ctx:,} tokens</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Two-column layout: Left for Document & Info, Right for Q&A
col_doc, col_chat = st.columns([1, 1], gap="large")

with col_doc:
    st.subheader("1. Document Upload & Extraction")

    existing_data_files = []
    if os.path.isdir("data"):
        existing_data_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]

    source_type = "Upload New File"
    if existing_data_files:
        source_type = st.radio("File Source", ["Upload New File", "Choose from ./data folder"], horizontal=True)

    uploaded_file = None
    selected_data_file = None

    if source_type == "Upload New File":
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    else:
        selected_data_file = st.selectbox("Select file from ./data", existing_data_files)

    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            file_bytes = uploaded_file.read()
            with st.spinner("Extracting text from uploaded PDF..."):
                st.session_state.extracted_text = extract_pdf_text_from_bytes(file_bytes)
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.last_token_metrics = None
    elif selected_data_file:
        file_path = os.path.join("data", selected_data_file)
        if st.session_state.current_file_name != selected_data_file:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            with st.spinner("Extracting text from local PDF..."):
                st.session_state.extracted_text = extract_pdf_text_from_bytes(file_bytes)
                st.session_state.current_file_name = selected_data_file
                st.session_state.messages = []
                st.session_state.last_token_metrics = None

    if st.session_state.current_file_name and st.session_state.extracted_text:
        char_count = len(st.session_state.extracted_text)
        word_count = len(st.session_state.extracted_text.split())
        est_tokens = int(char_count / 4)

        st.markdown(
            f"""
            <div class="pdf-meta-box">
                <b>Active File:</b> {st.session_state.current_file_name}<br>
                <b>Characters:</b> {char_count:,} | <b>Words:</b> ~{word_count:,} | <b>Est. Document Tokens:</b> ~{est_tokens:,}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📖 View Raw Extracted Text", expanded=False):
            st.text_area(
                "Document Text",
                st.session_state.extracted_text,
                height=350,
                disabled=True,
            )
    else:
        st.info("👆 Upload or select a PDF document to begin.")

with col_chat:
    st.subheader("2. Ask & Inspect Tokens")

    st.caption("Quick Queries:")
    quick_cols = st.columns(3)
    preset_query = None
    if quick_cols[0].button("📋 Key Findings", use_container_width=True):
        preset_query = "Summarize the key clinical findings and diagnoses."
    if quick_cols[1].button("💊 Medications", use_container_width=True):
        preset_query = "List all medications, dosages, and instructions mentioned."
    if quick_cols[2].button("🧪 Lab Results", use_container_width=True):
        preset_query = "Extract all laboratory test results, values, and reference ranges."

    chat_container = st.container(height=380)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="color: #888; text-align: center; margin-top: 120px;">
                    No messages yet.<br>Ask a specific question about the document below to query the LLM and analyze token metrics.
                </div>
                """,
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a question about the document...")
    final_query = preset_query or user_prompt

    if final_query:
        if not st.session_state.extracted_text:
            st.warning("Please upload a PDF document first before querying.")
        else:
            st.session_state.messages.append({"role": "user", "content": final_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(final_query)

            # Query Ollama native /api/chat endpoint with streaming to get token stats and prompt caching
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    payload = {
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": selected_system_prompt},
                            {
                                "role": "user",
                                "content": f"Document content:\n\n{st.session_state.extracted_text}\n\nQuestion: {final_query}",
                            },
                        ],
                        "options": {
                            "num_ctx": num_ctx,
                        },
                        "keep_alive": "30m",
                        "stream": True,
                    }

                    endpoint = f"{ollama_host}/api/chat"
                    try:
                        start_time = time.time()
                        response = requests.post(endpoint, json=payload, stream=True, timeout=120)
                        
                        if response.status_code != 200:
                            err_msg = f"❌ Ollama HTTP {response.status_code}: {response.text}"
                            response_placeholder.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        else:
                            final_metrics = {}
                            for line in response.iter_lines():
                                if line:
                                    chunk = json.loads(line.decode("utf-8"))
                                    if "message" in chunk and "content" in chunk["message"]:
                                        full_response += chunk["message"]["content"]
                                        response_placeholder.markdown(full_response + "▌")
                                    
                                    # Final chunk contains statistics
                                    if chunk.get("done", False):
                                        final_metrics = {
                                            "total_duration": chunk.get("total_duration", 0),
                                            "load_duration": chunk.get("load_duration", 0),
                                            "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                                            "prompt_eval_cached_count": chunk.get("prompt_eval_cached_count", 0),
                                            "prompt_eval_duration": chunk.get("prompt_eval_duration", 0),
                                            "eval_count": chunk.get("eval_count", 0),
                                            "eval_duration": chunk.get("eval_duration", 0),
                                        }

                            response_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

                            if final_metrics:
                                st.session_state.last_token_metrics = final_metrics
                                st.rerun()

                    except Exception as e:
                        err_msg = f"❌ Could not connect to Ollama at {endpoint}: {str(e)}"
                        response_placeholder.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
