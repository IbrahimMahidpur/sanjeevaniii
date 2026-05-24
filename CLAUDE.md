# CLAUDE.md — Medical AI Chatbot

> This file is the authoritative specification for Claude Code.
> Read it fully before writing any code. Follow the build order strictly.

---

## Project Overview

A production-grade Medical AI Chatbot with:
- QLoRA fine-tuned Mistral-7B on medical corpora
- Hybrid RAG pipeline (dense + BM25 + RRF reranking) using Qdrant
- LangGraph multi-node stateful agent
- Multimodal inputs: X-ray images (LLaVA-Med), lab PDFs (PyMuPDF + OCR), voice (Whisper)
- Multilingual support: Hindi, English, Spanish, Arabic, French (fastText detect + NLLB-200 translate)
- Web UI channel + REST API only (**NO WhatsApp, NO SMS**)
- FastAPI async gateway with Redis sessions and rate limiting
- vLLM serving, Docker Compose, Prometheus + Grafana monitoring

**Target metrics:** ≥90% domain accuracy · <2% hallucination · <2s response latency · 50+ concurrent users

---

## EXCLUDED FEATURES — DO NOT BUILD

- ❌ WhatsApp Business API integration (`channels/whatsapp.py`)
- ❌ SMS / Twilio / MSG91 integration (`channels/sms.py`)
- ❌ Any webhook handler for WhatsApp or SMS
- ❌ DLT registration or Twilio credentials

The `channels/` directory should exist but contain only `web.py` (SSE streaming for React frontend).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM base model | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | QLoRA via PEFT + bitsandbytes (4-bit NF4) |
| LLM serving | vLLM (OpenAI-compatible endpoint) |
| Agent framework | LangGraph (StateGraph) |
| Vector DB | Qdrant (local Docker) |
| Embeddings | BAAI/bge-large-en-v1.5 + bge-m3 (multilingual) |
| BM25 | rank_bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| Multimodal | LLaVA-Med (images), PyMuPDF + Tesseract (PDF), Whisper-large-v3 (voice) |
| Translation | facebook/nllb-200-distilled-1.3B |
| Language detect | fastText lid.176.bin |
| API | FastAPI (async) |
| Sessions | Redis |
| Web UI | React + Tailwind + SSE streaming |
| Monitoring | Prometheus + Grafana + LangSmith |
| Containers | Docker Compose |
| PII scrubbing | presidio-analyzer |

---

## Project Structure

```
medical-chatbot/
├── CLAUDE.md                  ← this file
├── .env.example
├── requirements.txt
├── README.md
│
├── data/
│   ├── scripts/
│   │   ├── pubmed_scraper.py      # PubMed API → JSONL
│   │   ├── pdf_extractor.py       # Extract text from medical PDFs
│   │   └── build_dataset.py       # Merge, deduplicate, format to train/val/test splits
│   ├── raw/                       # gitignored
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── train/
│   ├── qlora_train.py             # QLoRA fine-tuning entry point
│   ├── config.yaml                # Hyperparameters
│   └── merge_lora.py              # Merge LoRA adapter → base model
│
├── eval/
│   ├── run_eval.py                # MedQA benchmark runner
│   └── metrics.py                 # Hallucination rate, BERTScore, F1
│
├── rag/
│   ├── ingest.py                  # Chunk → embed → upsert to Qdrant
│   ├── retriever.py               # HybridRetriever: dense + BM25 + RRF
│   └── prompt_builder.py          # Build final prompt with retrieved context
│
├── agent/
│   ├── graph.py                   # LangGraph StateGraph definition
│   ├── state.py                   # MedicalState TypedDict
│   └── nodes/
│       ├── intent.py              # Classify query type
│       ├── retrieve.py            # Call RAG retriever
│       ├── generate.py            # Call vLLM
│       ├── validate.py            # Hallucination guard + disclaimer
│       └── respond.py             # Format and return response
│
├── multimodal/
│   ├── image_parser.py            # LLaVA-Med wrapper
│   ├── pdf_parser.py              # PyMuPDF + Tesseract + Camelot
│   └── voice_parser.py            # Whisper async transcription
│
├── multilingual/
│   ├── detector.py                # fastText language detection
│   └── translator.py              # NLLB-200 translate wrapper
│
├── channels/
│   └── web.py                     # SSE streaming responses for React UI
│
├── api/
│   ├── main.py                    # FastAPI app
│   ├── middleware.py              # Rate limiting, PII scrubbing, auth
│   └── routers/
│       ├── chat.py                # POST /chat, GET /chat/stream (SSE)
│       └── health.py              # GET /health, GET /ready
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   └── VoiceRecorder.tsx
│   │   └── hooks/
│   │       └── useSSEStream.ts
│   ├── package.json
│   └── tailwind.config.js
│
├── infra/
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── prometheus/
│       └── prometheus.yml
│
└── tests/
    ├── unit/
    │   ├── test_retriever.py
    │   ├── test_agent_nodes.py
    │   └── test_multilingual.py
    └── integration/
        ├── test_api.py
        └── test_full_pipeline.py
```

---

## Build Order — Follow This Exactly

### Phase 1 — Repository Scaffold & Data Pipeline

**Goal:** Scaffold the full project layout and build working data collection scripts.

1. Create the entire directory structure above with empty `__init__.py` files where needed.
2. Create `.env.example` with all required environment variables:
   ```
   VLLM_BASE_URL=http://localhost:8000/v1
   QDRANT_URL=http://localhost:6333
   REDIS_URL=redis://localhost:6379
   LANGSMITH_API_KEY=
   HF_TOKEN=
   SECRET_KEY=
   ```
3. Create `requirements.txt` with pinned versions for: torch, transformers, peft, bitsandbytes, trl, langgraph, langchain, langchain-community, qdrant-client, fastembed, rank-bm25, sentence-transformers, vllm, fastapi, uvicorn, redis, httpx, pytest, ruff, whisper, pymupdf, pytesseract, camelot-py, presidio-analyzer, fasttext, prometheus-client, langsmith.
4. Implement `data/scripts/pubmed_scraper.py`:
   - Use NCBI Entrez API (BioPython) to fetch abstracts for queries: "diabetes treatment", "hypertension management", "drug interactions", "symptoms diagnosis", etc.
   - Output: one JSONL per query, fields: `{"id": ..., "title": ..., "abstract": ..., "source": "pubmed"}`
5. Implement `data/scripts/pdf_extractor.py`:
   - Accept a directory of medical PDF files as input
   - Extract clean text using PyMuPDF, fall back to Tesseract OCR for scanned pages
   - Output JSONL with `{"id":..., "text":..., "source": "pdf", "filename": ...}`
6. Implement `data/scripts/build_dataset.py`:
   - Read all JSONL from `data/raw/`
   - Deduplicate by content hash
   - Format each record as an instruction-tuning pair: `{"instruction": "...", "input": "...", "output": "..."}`
   - Split 90/5/5 into `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
   - Print dataset statistics at end

**Verification:** `python data/scripts/build_dataset.py --sources pubmed --output data/processed/` should run without errors and produce the three split files.

---

### Phase 2 — QLoRA Fine-tuning

**Goal:** Fine-tune Mistral-7B-Instruct-v0.3 on the medical corpus using QLoRA.

1. Implement `train/config.yaml`:
   ```yaml
   model_name: mistralai/Mistral-7B-Instruct-v0.3
   output_dir: ./checkpoints
   lora_r: 16
   lora_alpha: 32
   lora_dropout: 0.05
   target_modules: [q_proj, v_proj, k_proj, o_proj]
   bits: 4
   bf16: true
   per_device_train_batch_size: 4
   gradient_accumulation_steps: 4
   learning_rate: 2e-4
   num_train_epochs: 3
   warmup_ratio: 0.03
   logging_steps: 10
   save_steps: 100
   eval_steps: 100
   max_seq_length: 2048
   ```
2. Implement `train/qlora_train.py`:
   - Load config from `config.yaml`
   - Load model with `BitsAndBytesConfig` (4-bit NF4, double quant)
   - Apply LoRA via `get_peft_model` with `LoraConfig`
   - Use HuggingFace `SFTTrainer` from `trl`
   - Load `data/processed/train.jsonl` and `val.jsonl`
   - Format each sample as Mistral chat format: `[INST] {instruction}\n{input} [/INST] {output}`
   - Evaluate on val set every `eval_steps`; log to LangSmith if `LANGSMITH_API_KEY` is set
3. Implement `train/merge_lora.py`:
   - Load base model + LoRA adapter from checkpoint
   - Merge with `model.merge_and_unload()`
   - Save merged model to `./merged_model/`
   - Verify by running a test inference

**Verification:** `python train/qlora_train.py --dry-run` should load the model, run one step, and exit cleanly.

---

### Phase 3 — RAG Pipeline

**Goal:** Build a hybrid retrieval system backed by Qdrant.

1. Implement `rag/ingest.py`:
   - Load `data/processed/train.jsonl`
   - Chunk text at 512 tokens with 50-token overlap using `langchain.text_splitter.RecursiveCharacterTextSplitter`
   - Embed chunks using `BAAI/bge-large-en-v1.5` (via `sentence-transformers`)
   - Also embed with `BAAI/bge-m3` for multilingual chunks (detect language with fastText first)
   - Upsert into Qdrant collection `medical_docs` with payload: `{text, source, chunk_id, language}`
   - Build and persist a BM25 index to `rag/bm25_index.pkl`
   - Print ingestion stats when done

2. Implement `rag/retriever.py` — `HybridRetriever` class:
   - `__init__`: load Qdrant client, load BM25 index, load cross-encoder reranker
   - `retrieve(query: str, top_k: int = 5) -> list[dict]`:
     - Dense retrieval: embed query with bge-large → Qdrant search, get top 20
     - Sparse retrieval: BM25 search, get top 20
     - Merge with Reciprocal Rank Fusion (RRF, k=60)
     - Rerank merged results with cross-encoder `cross-encoder/ms-marco-MiniLM-L-12-v2`
     - Return top `top_k` with fields: `{text, score, source}`
   - Handle empty results gracefully

3. Implement `rag/prompt_builder.py` — `build_medical_prompt(query, retrieved_docs, language="en") -> str`:
   - Format system prompt with medical persona and safety disclaimers
   - Append retrieved context labeled `[Context 1]`, `[Context 2]`, etc.
   - Append user query
   - Always include at the end: `"Note: Always consult a qualified doctor for medical advice."`

**Verification:** Start Qdrant via Docker, run `python rag/ingest.py`, then `python -c "from rag.retriever import HybridRetriever; r = HybridRetriever(); print(r.retrieve('diabetes symptoms'))"` should return 5 results.

---

### Phase 4 — LangGraph Agent

**Goal:** Build the stateful multi-turn agent graph.

1. Implement `agent/state.py` — `MedicalState(TypedDict)`:
   ```python
   messages: list[BaseMessage]
   query: str
   intent: str           # symptom_check | drug_info | lab_report | general | emergency
   retrieved_docs: list[dict]
   generated_response: str
   validated_response: str
   user_language: str    # ISO 639-1 code
   multimodal_context: str | None
   session_id: str
   error: str | None
   ```

2. Implement each node in `agent/nodes/`:

   **`intent.py` — `intent_node(state)`:**
   - Call vLLM to classify intent into: `symptom_check`, `drug_info`, `lab_report`, `general`, `emergency`
   - If `emergency` keywords detected (chest pain, stroke, seizure, etc.), set intent to `emergency`
   - Update `state.intent`

   **`retrieve.py` — `retrieve_node(state)`:**
   - Call `HybridRetriever.retrieve(state.query)`
   - If `state.multimodal_context` exists, prepend it to the query
   - Update `state.retrieved_docs`

   **`generate.py` — `generate_node(state)`:**
   - Build prompt via `prompt_builder.build_medical_prompt`
   - Call vLLM async (httpx to `VLLM_BASE_URL/v1/chat/completions`)
   - Stream tokens; collect full response
   - Update `state.generated_response`

   **`validate.py` — `validate_node(state)`:**
   - Check response does not contradict retrieved docs (basic overlap check)
   - If intent is `emergency`: prepend "⚠️ EMERGENCY: Call emergency services immediately. " to the response
   - Always append: "\n\n*This is AI-generated information. Please consult a qualified doctor.*"
   - Check for PII in response using presidio; redact if found
   - Update `state.validated_response`

   **`respond.py` — `respond_node(state)`:**
   - If `state.user_language != "en"`: translate `state.validated_response` via NLLB
   - Preserve drug names, ICD codes, and dosage units (regex list) from translation
   - Append final message to `state.messages`
   - Return updated state

3. Implement `agent/graph.py` — `build_graph() -> CompiledGraph`:
   - Create `StateGraph(MedicalState)`
   - Add nodes: `intent`, `retrieve`, `generate`, `validate`, `respond`
   - Add edges: `intent → retrieve → generate → validate → respond → END`
   - Add conditional edge from `intent`: if `emergency` → skip `retrieve`, go directly to `generate`
   - Compile and return graph

**Verification:** `python -c "from agent.graph import build_graph; g = build_graph(); print(g.get_graph().draw_ascii())"` should print the graph topology.

---

### Phase 5 — Multimodal Support

**Goal:** Parse images, PDFs, and voice input before the agent processes them.

1. Implement `multimodal/image_parser.py` — `parse_image(image_bytes: bytes) -> str`:
   - Load LLaVA-Med (or `llava-hf/llava-1.5-7b-hf` as fallback if LLaVA-Med unavailable)
   - Preprocess image: resize to 336×336, normalize
   - Prompt: `"You are a medical image analyst. Describe all clinical findings in this medical image. Be specific about abnormalities, measurements, and anatomical structures."`
   - Return structured text: `{"type": "medical_image", "findings": "...", "confidence": 0.0-1.0}`
   - Handle JPEG, PNG, DICOM (pydicom) formats

2. Implement `multimodal/pdf_parser.py` — `parse_pdf(pdf_bytes: bytes) -> str`:
   - Try digital text extraction with PyMuPDF first
   - If page has <50 words, fall back to Tesseract OCR on rasterized page (300 DPI)
   - Extract tables using Camelot (lattice mode for structured lab reports)
   - Return structured text with sections: `[PATIENT INFO]`, `[LAB VALUES]`, `[FINDINGS]`, `[IMPRESSION]`
   - Identify lab values with units and flag abnormal ranges (e.g., glucose > 126 mg/dL)

3. Implement `multimodal/voice_parser.py` — `async parse_voice(audio_bytes: bytes) -> dict`:
   - Load `openai/whisper-large-v3`
   - Transcribe audio
   - Return `{"transcript": "...", "language": "hi", "confidence": 0.95}`
   - Support formats: WAV, MP3, OGG, WebM
   - Run async in executor to avoid blocking

**Verification:** Unit test each parser with a sample file in `tests/unit/test_multimodal.py`.

---

### Phase 6 — Multilingual Support

**Goal:** Detect language and translate seamlessly through the pipeline.

1. Implement `multilingual/detector.py` — `detect_language(text: str) -> str`:
   - Download `lid.176.bin` from fastText if not present
   - Return ISO 639-1 code (e.g., `"hi"`, `"en"`, `"es"`)
   - Cache the model in memory after first load
   - Handle short texts (<10 chars) by defaulting to `"en"`

2. Implement `multilingual/translator.py` — `Translator` class:
   - Load `facebook/nllb-200-distilled-1.3B` (CPU for small batches, GPU if available)
   - `translate(text: str, src_lang: str, tgt_lang: str) -> str`
   - Before translation, extract and preserve: drug names (regex from RxNorm list), ICD-10 codes, dosage patterns (`\d+\s*mg/kg`)
   - After translation, restore preserved terms back to their original forms
   - Language codes: use NLLB format (e.g., `hin_Deva` for Hindi, `spa_Latn` for Spanish)
   - Cache translated results in Redis with 1-hour TTL

3. Update `agent/nodes/intent.py` to call `detect_language` and store in `state.user_language`
4. Update `agent/nodes/respond.py` to call `Translator` when `state.user_language != "en"`

**Verification:** `python -c "from multilingual.translator import Translator; t = Translator(); print(t.translate('Take 500mg paracetamol twice daily', 'en', 'hi'))"` — drug name and dosage should remain in English.

---

### Phase 7 — FastAPI Gateway + Web Channel

**Goal:** Build the async API and SSE streaming for the React frontend. **No WhatsApp or SMS.**

1. Implement `api/middleware.py`:
   - `RateLimitMiddleware`: 10 requests/minute per session_id, backed by Redis sliding window
   - `PIIScrubMiddleware`: run `presidio-analyzer` on request body; replace PII with `[REDACTED]` before logging
   - `AuthMiddleware`: validate `Authorization: Bearer <token>` header (simple shared secret from `.env` for now)

2. Implement `api/routers/health.py`:
   - `GET /health` → `{"status": "ok", "timestamp": ...}`
   - `GET /ready` → check Qdrant, Redis, vLLM connectivity; return 200 or 503

3. Implement `api/routers/chat.py`:
   - `POST /chat` — sync endpoint:
     - Accept: `{"message": str, "session_id": str, "file": optional base64}`
     - Detect MIME type of `file` if present; route to appropriate parser
     - Retrieve or create session from Redis
     - Run `agent_graph.ainvoke(state)`
     - Return: `{"response": str, "session_id": str, "intent": str}`
   - `GET /chat/stream` — SSE endpoint:
     - Same input via query params or headers
     - Stream tokens from vLLM as `data: {"token": "..."}` events
     - Send `data: [DONE]` when complete

4. Implement `channels/web.py` — `WebChannel`:
   - Helper to format agent responses as SSE events
   - Strip markdown for plain-text fallback

5. Implement `api/main.py`:
   - Create FastAPI app with lifespan (startup: connect Redis + Qdrant; shutdown: cleanup)
   - Add CORS middleware (allow React dev server origin)
   - Register routers: `/` health, `/api/v1/` chat
   - Add middleware stack: PII scrub → rate limit → auth

**Verification:** `uvicorn api.main:app --reload` should start; `GET /health` returns 200; `POST /api/v1/chat` with `{"message": "What is diabetes?", "session_id": "test-1"}` returns a response.

---

### Phase 8 — React Web Frontend

**Goal:** Chat UI with streaming, file upload, and voice input.

1. Scaffold React app in `frontend/` using Vite:
   ```
   npm create vite@latest frontend -- --template react-ts
   cd frontend && npm install tailwindcss @tailwindcss/typography axios
   ```

2. Implement `frontend/src/hooks/useSSEStream.ts`:
   - Connect to `GET /api/v1/chat/stream`
   - Append tokens to `streamingText` state as they arrive
   - Handle `[DONE]` event; handle errors

3. Implement `frontend/src/components/ChatWindow.tsx`:
   - Message history list
   - Input box with send button
   - SSE streaming: show typing indicator, update message bubble token-by-token
   - Scroll to bottom on new message

4. Implement `frontend/src/components/MessageBubble.tsx`:
   - User vs assistant styling
   - Render markdown (use `react-markdown`)
   - Show intent badge (e.g., 🔴 Emergency, 💊 Drug Info, 🔬 Lab Report)
   - Show disclaimer footer on assistant messages

5. Implement `frontend/src/components/FileUpload.tsx`:
   - Drag-and-drop + click to upload
   - Accept: `image/*`, `application/pdf`
   - Convert to base64, attach to next message
   - Preview thumbnail for images

6. Implement `frontend/src/components/VoiceRecorder.tsx`:
   - Use `MediaRecorder` API to record from microphone
   - Export as WebM blob → base64
   - Show recording indicator with timer

7. `frontend/src/App.tsx`:
   - Assemble all components
   - Dark/light mode toggle
   - Language selector (EN/HI/ES/AR/FR) — sets `Accept-Language` header

**Verification:** `cd frontend && npm run dev` → chat UI loads; can send a message and see streamed response.

---

### Phase 9 — Infrastructure & Monitoring

**Goal:** Docker Compose for all services, Prometheus metrics, Grafana dashboard.

1. Implement `infra/docker-compose.yml` with services:
   ```yaml
   services:
     api:          # FastAPI, port 8080, replicas: 1 (or 3 behind nginx)
     vllm:         # vLLM server, port 8000, GPU device 0, model: ./merged_model
     qdrant:       # qdrant/qdrant:latest, port 6333
     redis:        # redis:7-alpine, port 6379
     postgres:     # postgres:15-alpine (for audit logs), port 5432
     prometheus:   # prom/prometheus, port 9090
     grafana:      # grafana/grafana, port 3001
     frontend:     # nginx serving built React app, port 3000
   ```
   - All services on a shared `medchat` Docker network
   - Persistent volumes for: qdrant data, postgres data, prometheus data, grafana data, model weights
   - Health checks for all services

2. Add Prometheus metrics to `api/main.py`:
   - `http_requests_total` (counter, labels: method, endpoint, status)
   - `http_request_duration_seconds` (histogram)
   - `llm_response_latency_seconds` (histogram)
   - `rag_retrieval_latency_seconds` (histogram)
   - `active_sessions` (gauge)
   - Expose `/metrics` endpoint

3. Create `infra/prometheus/prometheus.yml` scraping the FastAPI `/metrics` endpoint.

4. Create a Grafana dashboard JSON at `infra/grafana/dashboard.json` with panels:
   - Request rate over time
   - p50/p95/p99 response latency
   - Active sessions
   - LLM latency vs RAG latency
   - Error rate

5. Implement GitHub Actions CI in `.github/workflows/ci.yml`:
   - Trigger: push to `main`, PR to `main`
   - Jobs: lint (ruff), unit tests (pytest tests/unit/), build Docker image
   - Cache pip dependencies and Docker layers

**Verification:** `docker compose up -d` should bring up all services; `http://localhost:3000` loads the chat UI; `http://localhost:9090` shows Prometheus targets all UP.

---

### Phase 10 — Tests & Evaluation

**Goal:** Unit tests, integration tests, and medical accuracy evaluation.

1. Write unit tests in `tests/unit/`:
   - `test_retriever.py`: mock Qdrant, verify RRF merge logic
   - `test_agent_nodes.py`: test each LangGraph node in isolation with mock LLM
   - `test_multilingual.py`: verify translation preserves drug names
   - `test_multimodal.py`: test PDF and voice parsers with sample files

2. Write integration tests in `tests/integration/`:
   - `test_api.py`: spin up FastAPI with TestClient; test `/health`, `/chat`, `/chat/stream`
   - `test_full_pipeline.py`: end-to-end test with a sample medical question; verify response contains disclaimer

3. Implement `eval/run_eval.py`:
   - Download MedQA-USMLE test set (1273 questions)
   - For each question: run through full agent pipeline
   - Compute accuracy (correct answer match)
   - Print summary: accuracy %, avg latency, hallucination rate estimate
   - Target: ≥90% accuracy

4. Implement `eval/metrics.py`:
   - `compute_bertscore(predictions, references)` using `bert_score` library
   - `estimate_hallucination_rate(responses, retrieved_docs)`: check what fraction of factual claims in responses are not grounded in retrieved docs (keyword overlap heuristic)
   - `compute_f1(predictions, references)`: token-level F1

**Verification:** `pytest tests/unit/ -v` should pass all tests; `pytest tests/integration/ -v` should pass with services running.

---

## Coding Standards

- All Python code must pass `ruff check` with zero errors
- All async functions must use `async/await` — no blocking I/O in async context
- All modules must have docstrings
- Type hints required on all function signatures
- Environment variables accessed only through `os.getenv(KEY)` — never hardcoded
- Secrets never committed to git (enforced by `.gitignore` and `.env.example`)
- Every function that calls the LLM must have a timeout (default 30s)
- Medical disclaimer must appear on EVERY response — never skip this

## Safety Requirements

- PII scrubbing runs on every incoming request before logging
- No medical images stored permanently without explicit consent flow
- Rate limiting enforced per session (10 req/min)
- Emergency intent detected → prepend emergency warning → always recommend calling emergency services
- All responses append: *"This is AI-generated information. Please consult a qualified doctor."*

## Environment Variables (copy to `.env`)

```
VLLM_BASE_URL=http://localhost:8000/v1
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://meduser:medpass@localhost:5432/medchat
LANGSMITH_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token
SECRET_KEY=your_random_secret_key_here
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## Running the Project (Quick Start)

```bash
# 1. Clone and set up environment
git clone <repo> && cd medical-chatbot
cp .env.example .env  # fill in your keys
pip install -r requirements.txt

# 2. Build dataset
python data/scripts/build_dataset.py --sources pubmed --output data/processed/

# 3. Fine-tune (skip if using base Mistral directly)
python train/qlora_train.py
python train/merge_lora.py

# 4. Start infrastructure
docker compose up -d qdrant redis postgres prometheus grafana

# 5. Ingest documents into RAG
python rag/ingest.py

# 6. Start vLLM
docker compose up -d vllm

# 7. Start API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# 8. Start frontend
cd frontend && npm install && npm run dev
```

---

*Built with Claude Code. Channels: Web UI + REST API only. No WhatsApp or SMS.*
