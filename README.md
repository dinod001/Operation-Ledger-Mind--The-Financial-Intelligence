# 🧠 Operation Ledger-Mind: The Financial Intelligence
### AI Engineer Essentials - Mini Project 01 (Weeks 01-03)

![Project Banner](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logoColor=white)
![Tech Stack](https://img.shields.io/badge/Stack-Advanced_RAG_|_Fine--Tuning_|_LLM_Judge-blue?style=for-the-badge)

## 🎯 The Mission
As the Lead AI Architect for **Alpha-Yield Capital**, I have developed a dual-strategy system to extract deep insights from Uber Technologies' 2024 Annual Report. This project conducts a "Showdown" between parametric memory (Fine-tuning) and non-parametric memory (RAG) to determine the superior architecture for financial analysis.

---

## 🏛️ Project Architecture

The project is structured into four core stages, each building upon the previous one to create a comprehensive evaluation ecosystem.

### 🏭 Part 1: The Data Factory (`01_data_factory.ipynb`)
Financial PDFs are complex. We transform raw data into a structured instruction-tuning dataset.
- **Workflow**: Ingestion -> Clean Chunking (1500 chars) -> Synthetic Q&A Generation.
- **Generation**: Utilizes a dual-LLM pipeline (Question Gen via LLM A, Answer Gen via LLM B) for high faithfulness.
- **Output**: 80/20 split resulting in `train.jsonl` and `golden_test_set.jsonl`.

### 🤖 Part 2: "The Intern" (Fine-Tuning) (`02_finetuning_intern.ipynb`)
Teaching a base model (Llama-3) Uber’s specific strategy and stylistic tone.
- **Infrastructure**: LoRA (Low-Rank Adaptation) + 4-bit Quantization (bitsandbytes).
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- **Trainer**: `SFTTrainer` with a minimum of 100 steps.
- **Capability**: Integrated `query_intern(question)` inference pipeline.

### 📚 Part 3: "The Librarian" (Advanced RAG) (`03_rag_librarian.ipynb`)
A high-precision retrieval system for exact page citations and hard financial facts.
- **Vector DB**: Weaviate (Local instance).
- **Hybrid Retrieval**: Dense Vector Search + BM25 Keyword Search.
- **Refinement**: Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking for top-tier relevance.
- **Capability**: Integrated `query_librarian(question)` function.

### ⚔️ Part 4: The Showdown (Evaluation) (`04_evaluation_arena.ipynb`)
A ruthless benchmark comparing accuracy, cost, and latency.
- **Metrics**: 
    - **ROUGE-L**: Textual overlap with ground truth.
    - **LLM-as-a-Judge**: Faithfulness & Accuracy scoring (1-5) using a reasoning model.
    - **Latency**: Generation time performance tracking.
- **Bonus**: 30-day cloud cost estimation for enterprise-scale serving.

---

## 📁 Repository Structure
```bash
├── notebooks/
│   ├── 01_data_factory.ipynb       # PDF Processing & Synthetic Data
│   ├── 02_finetuning_intern.ipynb   # QLoRA Training Loop
│   ├── 03_rag_librarian.ipynb       # Advanced RAG Pipeline
│   └── 04_evaluation_arena.ipynb    # Metrics & Comparison
├── src/
│   ├── config/
│   │   └── config.yaml              # Central Configuration (Paths, Models, Params)
│   ├── services/
│   │   └── llm_services.py          # Unified LLM provider wrappers
│   └── utils/
│       └── cost_tracker.py          # Token & Financial cost management
├── artifacts/                       # Generated datasets and eval results
├── pyproject.toml                   # Project metadata & build config
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Weaviate (Local Docker or Embedded)
- OpenAI / Gemini / Groq API Keys (configured in `.env`)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Control all model parameters and file paths through `src/config/config.yaml`.

---

## 📊 Final Results Preview
*Summary of results available in the Evaluation Arena report.*
- **Librarian (RAG)**: Superior for exact fact-finding and citations.
- **Intern (FT)**: Superior for stylistic consistency and summarizing general sentiment.

---

**Developed for Alpha-Yield Capital | Mini Project 01 Submission**
