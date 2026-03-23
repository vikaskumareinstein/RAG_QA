# Document Q&A

A RAG-based document question-answering system powered by Google Gemini.
Upload up to 3 PDF documents and ask natural language questions — answers include citations showing which document and page the information came from.

---

## Setup & Run

### 1. Install uv (if not already installed) 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh 
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run

```bash
uv run streamlit run app.py   
```

The app opens at `http://localhost:8501`.

---


### Key decisions

- **No RAG framework** 
- **Pydantic `BaseSettings`**
- **In-memory ChromaDB**
- **Separate embed task types**



## Known Limitations

- **In-memory index**
- **Character-based chunking**
- **English language Only**
- **Scanned/Image PDFs Not supported**

