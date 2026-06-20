# Chat with Websites 🤖

A Streamlit application that lets you chat with the content of any website. It scrapes the provided URL, indexes the content using embeddings, and uses a RAG (Retrieval-Augmented Generation) pipeline powered by Groq's LLaMA 3.3 model to answer your questions based on that content.

## Features

- 🌐 Load and parse content from any website URL
- 🔍 Semantic search using FAISS vector store and HuggingFace embeddings
- 💬 Conversational chat interface with message history
- ⚡ Fast LLM responses via Groq's `llama-3.3-70b-versatile` model

## Tech Stack

- [Streamlit](https://streamlit.io/) – web app interface
- [LangChain](https://www.langchain.com/) – orchestration framework
- [FAISS](https://github.com/facebookresearch/faiss) – vector similarity search
- [HuggingFace Embeddings](https://huggingface.co/) – `all-MiniLM-L6-v2` model
- [Groq](https://groq.com/) – LLM inference (`llama-3.3-70b-versatile`)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with the following keys:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

- Get a Groq API key from [console.groq.com](https://console.groq.com/)
- Get a HuggingFace API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Enter a website URL in the sidebar.
2. Wait for the content to be loaded and indexed.
3. Type your question in the chat input and get answers based on the website's content.

## Requirements

```txt
streamlit
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
python-dotenv
beautifulsoup4
```

## How It Works

1. **Document Loading**: `WebBaseLoader` fetches and parses the HTML content of the given URL.
2. **Chunking**: `RecursiveCharacterTextSplitter` splits the content into smaller chunks for embedding.
3. **Vector Store**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in a FAISS index.
4. **Retrieval**: On each user query, the most relevant chunks are retrieved from the vector store.
5. **Generation**: The retrieved context, chat history, and user query are passed to the Groq LLM to generate a response.

 License

This project is licensed under the MIT License.
