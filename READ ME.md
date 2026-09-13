# 📚 RAG Document Q&A with Ollama Embeddings + Groq

## 📌 Project Overview

**RAG Document Q&A** is an AI-powered question-answering application that allows users to ask questions about research papers stored as PDF documents.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve the most relevant information from the uploaded research papers and then uses a **Groq-hosted LLM** to generate an accurate answer based on the retrieved context.

The project combines **Streamlit, LangChain, Ollama, FAISS, and Groq** to create an interactive document question-answering system.

---

## 🎯 Problem Statement

Reading and understanding multiple research papers can be time-consuming. Users often need to search through hundreds of pages to find specific information.

Traditional keyword-based searching may not understand the meaning or context of a user's question.

For example, a user may have several research papers and want to ask:

* What is the main objective of this research?
* What methodology was used?
* What are the key findings?
* What dataset was used?
* What are the limitations of the proposed approach?
* How does the proposed method compare with existing methods?

Manually searching through each PDF to answer these questions is inefficient.

### 💡 Proposed Solution

This project provides a **RAG-based document question-answering system** that:

1. Loads research papers from a PDF folder.
2. Extracts text from the PDFs.
3. Splits the text into smaller chunks.
4. Converts the chunks into vector embeddings using Ollama.
5. Stores the embeddings in a FAISS vector database.
6. Retrieves the most relevant document chunks for a user's question.
7. Sends the retrieved context to a Groq LLM.
8. Generates an answer based only on the retrieved documents.
9. Displays the answer and the source document content.

---

# 🧠 What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

Instead of asking an LLM to answer a question only from its pre-trained knowledge, RAG first retrieves relevant information from a user's documents.

The overall workflow is:

```text
                 Research Papers
                       │
                       ▼
                 PDF Document Loader
                       │
                       ▼
                 Extract Text
                       │
                       ▼
                 Text Splitting
                       │
                       ▼
              Ollama Embeddings
                       │
                       ▼
                 FAISS Vector DB
                       │
                       │
User Question ────────┤
                       ▼
                  Retriever
                       │
                       ▼
             Relevant Document Chunks
                       │
                       ▼
                  Groq LLM
                       │
                       ▼
                   Answer
```

---

# 🚀 Key Features

### 📄 PDF Document Processing

The application loads PDF documents from the:

```text
research_papers/
```

directory using LangChain's `PyPDFDirectoryLoader`.

---

### ✂️ Text Chunking

Large PDF documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

This makes document retrieval more efficient.

---

### 🧠 Ollama Embeddings

The project uses the Ollama embedding model:

```text
nomic-embed-text
```

The document chunks are converted into numerical vector representations.

---

### 🔎 FAISS Vector Search

The generated embeddings are stored using **FAISS**.

FAISS allows the application to efficiently find document chunks that are semantically similar to the user's question.

The application retrieves the top 4 relevant chunks:

```python
search_kwargs={"k": 4}
```

---

### 🤖 Groq LLM

The retrieved document context is passed to a Groq-hosted language model:

```python
ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)
```

The model generates the final answer using the retrieved context.

---

### 🛡️ Context-Based Answers

The prompt instructs the LLM to answer **only from the provided document context**.

If the information cannot be found in the retrieved documents, the application instructs the model to respond:

```text
I don't know based on the provided documents.
```

This helps reduce unsupported answers.

---

### ⏱️ Response Time

The application measures how long the RAG pipeline takes to generate an answer.

Example:

```text
Response time: 2.35 seconds
```

---

### 📑 Document Similarity Search

The application also displays the retrieved document chunks in an expandable section.

This allows users to inspect the information used to generate the answer.

---

# 🛠️ Technology Stack

| Technology           | Purpose                             |
| -------------------- | ----------------------------------- |
| **Python**           | Programming language                |
| **Streamlit**        | Web application interface           |
| **LangChain**        | RAG application framework           |
| **LangChain Groq**   | Integration with Groq LLM           |
| **Groq**             | Large Language Model inference      |
| **Ollama**           | Local embedding generation          |
| **nomic-embed-text** | Embedding model                     |
| **FAISS**            | Vector database / similarity search |
| **PyPDF**            | PDF document processing             |
| **python-dotenv**    | Environment variable management     |

---

# 🏗️ Project Architecture

The application consists of the following major components:

### 1. Document Loader

```python
PyPDFDirectoryLoader("research_papers")
```

Loads PDF files from the `research_papers` directory.

### 2. Text Splitter

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Splits the extracted document text into manageable chunks.

### 3. Embedding Model

```python
OllamaEmbeddings(
    model="nomic-embed-text"
)
```

Converts text chunks into vector embeddings.

### 4. Vector Database

```python
FAISS.from_documents(...)
```

Stores the embeddings and enables similarity-based retrieval.

### 5. Retriever

```python
vectors.as_retriever(
    search_kwargs={"k": 4}
)
```

Retrieves the four most relevant chunks for the user's question.

### 6. Document Chain

```python
create_stuff_documents_chain(
    llm,
    prompt
)
```

Combines the retrieved documents with the prompt and sends them to the LLM.

### 7. Retrieval Chain

```python
create_retrieval_chain(
    retriever,
    document_chain
)
```

Connects document retrieval and answer generation into a single RAG pipeline.

---

# 📁 Project Structure

```text
4-RAG-Document-QA/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── .env
│
└── research_papers/
    ├── paper1.pdf
    ├── paper2.pdf
    └── ...
```

> ⚠️ The `.env` file should **not** be uploaded to GitHub because it contains the Groq API key.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
```

Move into the project directory:

```bash
cd rag-document-qa
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
```

Replace:

```text
your_groq_api_key
```

with your actual Groq API key.

### ⚠️ Security

Never commit the `.env` file to GitHub.

The `.gitignore` file should contain:

```gitignore
.env
```

---

# 🦙 Ollama Setup

This project uses Ollama for generating embeddings.

Install Ollama and download the required embedding model.

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

---

# 📄 Add Research Papers

Create a folder named:

```text
research_papers
```

Place your PDF research papers inside it.

Example:

```text
research_papers/
│
├── research_paper_1.pdf
├── research_paper_2.pdf
└── research_paper_3.pdf
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖥️ How to Use

### Step 1 — Start the application

Run:

```bash
streamlit run app.py
```

### Step 2 — Create document embeddings

Click:

```text
Create Document Embeddings
```

The application will:

```text
PDFs
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Ollama Embeddings
 ↓
FAISS Vector Database
```

### Step 3 — Ask a question

Enter a question about your research papers.

For example:

```text
What is the main objective of the research?
```

### Step 4 — View the answer

The Groq LLM generates an answer using the retrieved document context.

### Step 5 — View retrieved documents

Open:

```text
Document Similarity Search
```

to inspect the document chunks retrieved for your question.

---

# 🔄 RAG Workflow

The complete workflow is:

```text
                ┌───────────────────┐
                │   Research PDFs   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   PDF Loader      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │  Text Splitter    │
                │  Chunk = 1000     │
                │  Overlap = 200    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Ollama Embeddings │
                │ nomic-embed-text  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │       FAISS       │
                │   Vector Store    │
                └─────────┬─────────┘
                          │
                          │
                ┌─────────▼─────────┐
                │   User Question   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │    Retriever      │
                │    Top 4 Chunks   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │     Groq LLM      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Final Answer    │
                └───────────────────┘
```

---

# 💡 Example Questions

Users can ask questions such as:

```text
What is the main objective of this research paper?
```

```text
What methodology was used in the study?
```

```text
What dataset was used?
```

```text
What are the key findings?
```

```text
What are the limitations of the proposed approach?
```

```text
What technologies were used in this research?
```

```text
What is the conclusion of the paper?
```

---

# 🎯 Use Cases

This application can be useful for:

* 📚 Research paper analysis
* 🎓 Student projects
* 🔬 Academic research
* 📄 Document question answering
* 🧑‍💻 Technical documentation search
* 📖 Literature review
* 🏢 Internal document search
* 🤖 AI-powered knowledge assistants

---

# 🔐 Security Considerations

The Groq API key is stored using an environment variable:

```env
GROQ_API_KEY=your_api_key
```

The application reads the key using:

```python
os.getenv("GROQ_API_KEY")
```

API keys should never be hard-coded in the Python source code or committed to a public GitHub repository.

---

# 🚧 Current Limitations

The current version has some limitations:

* PDF documents must be placed in the `research_papers` folder.
* Users cannot upload PDFs directly through the Streamlit interface.
* Embeddings are generated when the user clicks the embedding button.
* The FAISS vector database is maintained in Streamlit session state rather than persisted to disk.
* The application currently retrieves the top 4 relevant chunks.
* Ollama needs to be installed and running locally.
* The application requires a valid Groq API key.

---

# 🔮 Future Enhancements

Possible improvements include:

* 📤 Upload PDFs directly through Streamlit
* 💾 Persist the FAISS vector database
* 💬 Add conversational chat history
* 📚 Support multiple document formats
* 🔎 Add source citations
* 🧠 Add hybrid search
* ⚡ Improve retrieval performance
* 👤 Add user authentication
* ☁️ Deploy the application to the cloud
* 📊 Add document and retrieval analytics
* 🗂️ Allow users to manage multiple document collections

---

# 📌 Learning Outcomes

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Vector embeddings
* Semantic search
* Vector databases
* Document processing
* LangChain
* Ollama
* Groq
* FAISS
* Streamlit
* Prompt engineering
* Environment variable management

---

# 👩‍💻 Author

**Sireesha Malla**

GitHub:
`https://github.com/YOUR_USERNAME`

---

# ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
