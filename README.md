# 📚 Papeer — Research Paper Assistant

Papeer is an advanced, AI-powered research assistant built with Streamlit, LangChain, and LangGraph. It is designed to help researchers, students, and professionals interact with research papers, verify scientific claims against the latest literature, and seamlessly search the web for supplementary context.

## 🌟 Key Features

- **Multi-Source Document Ingestion**: 
  - Upload local files (PDF, TXT, Markdown).
  - Scrape data directly from Web URLs.
  - Fetch papers directly from **ArXiv** using either the paper ID or title.
- **Intelligent Routing (LangGraph)**: The core engine uses a state machine to automatically determine if a user's question requires retrieving information from uploaded papers, verifying a claim, or answering from general knowledge.
- **Claim Verification**: Ask the assistant to fact-check a claim. It will query the live web and ArXiv to find the most recent literature and determine if the claim has been superseded or updated.
- **"By The Way" Side Channel**: Use the `/btw` command in the chat to ask off-topic or real-time questions (e.g., `/btw What is the current price of Bitcoin?`). This bypasses the main vector store and doesn't pollute your research chat history.
- **Session Management**: Automatically names your chat sessions based on your first message and maintains isolated vector spaces (multi-tenancy) so papers from one session don't bleed into another.
- **Cost-Optimized Embeddings**: Utilizes local hashing and caching for text embeddings to prevent duplicate API calls to Gemini when reloading the same documents.

---

## 📂 Project Structure

Here is a breakdown of the entire project and what each file/folder does:

### 🖥️ Frontend & Root Files
- **`app.py`**: The main entry point for the Streamlit web application. It handles the UI layout, file uploading, session state management, chat history rendering, and streams responses from the backend graph.
- **`pyproject.toml` & `uv.lock`**: Configuration files for project metadata and dependencies. The project uses `uv` for fast dependency resolution.
- **`requirements.txt`**: Standard list of Python dependencies.
- **`.env`** *(Not tracked in git)*: Stores sensitive API keys (`GEMINI_API_KEY`, `TAVILY_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`).

### ⚙️ `backend/` (Core Logic Engine)
This folder contains the LangChain and LangGraph logic.
- **`__init__.py`**: Marks the backend folder as a Python package.
- **`rag_graph.py`**: The "brain" of the app. Defines a directed state graph using LangGraph to orchestrate the flow: Routing -> Retrieval -> Relevancy Check -> Query Rewriting -> Fact Checking -> Answer Generation. Uses SQLite for memory checkpointing.
- **`vector_store.py`**: Manages the connection to the Qdrant vector database. Handles the embedding of document chunks (using Gemini) and performs cosine similarity searches. Implements `CacheBackedEmbeddings` to save costs.
- **`paper_loader.py`**: Contains specialized functions to download, parse, and chunk documents from ArXiv, raw URLs, or local files using LangChain loaders and RecursiveCharacterTextSplitters.
- **`btw_handler.py`**: A specialized, lightweight LangChain pipeline specifically for handling the `/btw` side-channel questions. It streams answers directly back to the UI.
- **`models.py`**: Defines strict Pydantic schemas (e.g., `RouterDecision`, `ClaimVerificationResult`) used to enforce structured JSON outputs from the LLMs.

### 📖 `backend_explanations/` (Documentation)
A dedicated folder containing highly detailed, line-by-line Markdown explanations of every Python file inside the `backend/` directory. Useful for onboarding and understanding the precise mechanics (like LCEL pipelines and decorators) used in the code.

### 📁 Other Folders
- **`documents/`**: A storage directory for sample or default research papers (e.g., `Openclaw_Research_Report.pdf`).
- **`src/papeer/`**: The source package directory for the application if installed as a module.
- **`embedding_cache/`** *(Generated at runtime)*: A local file store that caches Gemini embeddings to save money on duplicate texts.
- **`sessions.json`** *(Generated at runtime)*: Stores the metadata and titles of your chat sessions.
- **`checkpoints.db`** *(Generated at runtime)*: An SQLite database used by LangGraph to persist chat history and graph state.

---

## 🚀 Getting Started

### Prerequisites
- Python >= 3.14
- API Keys for **Gemini**, **Tavily** (for web search), and **Qdrant** (for vector storage).

### Installation
1. Clone the repository.
2. Install dependencies using `uv` (recommended) or `pip`:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY="your-key"
   TAVILY_API_KEY="your-key"
   QDRANT_URL="your-cluster-url"
   QDRANT_API_KEY="your-key"
   ```

### Running the App
Start the Streamlit server by running:
```bash
streamlit run app.py
```
Then, open the provided `localhost` URL in your browser. Use the sidebar to upload a paper, enter an ArXiv ID, or paste a URL, and start chatting!
