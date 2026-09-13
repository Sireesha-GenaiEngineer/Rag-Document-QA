import streamlit as st
import os
import time

from dotenv import load_dotenv

# LLM from Groq
from langchain_groq import ChatGroq

# Embeddings from Ollama
from langchain_ollama import OllamaEmbeddings

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chains
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# Vector database
from langchain_community.vectorstores import FAISS

# PDF loader
from langchain_community.document_loaders import PyPDFDirectoryLoader


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if not groq_api_key:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("RAG Document Q&A with Ollama Embeddings + Groq")

st.write(
    "Ask questions about your research papers using "
    "Ollama embeddings, FAISS, and a Groq Llama model."
)


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)

# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful research assistant.

    Answer the question based ONLY on the provided context.

    If the answer is not available in the context,
    say:

    "I don't know based on the provided documents."

    Give a clear, accurate and concise answer.

    <context>
    {context}
    </context>

    Question:
    {input}

    Answer:
    """
)


# ============================================================
# CREATE VECTOR EMBEDDINGS
# ============================================================

def create_vector_embedding():

    # Create vector database only once
    if "vectors" not in st.session_state:

        # ----------------------------------------------------
        # OLLAMA EMBEDDINGS
        # ----------------------------------------------------

        st.session_state.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # ----------------------------------------------------
        # LOAD PDF DOCUMENTS
        # ----------------------------------------------------

        st.session_state.loader = PyPDFDirectoryLoader(
            "research_papers"
        )

        st.session_state.docs = (
            st.session_state.loader.load()
        )

        # ----------------------------------------------------
        # CHECK WHETHER PDFs EXIST
        # ----------------------------------------------------

        if not st.session_state.docs:
            st.error(
                "No PDF documents were found in "
                "'research_papers' folder."
            )
            st.stop()

        # ----------------------------------------------------
        # TEXT SPLITTER
        # ----------------------------------------------------

        st.session_state.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        )

        # ----------------------------------------------------
        # SPLIT DOCUMENTS
        # ----------------------------------------------------

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
        )

        # ----------------------------------------------------
        # CREATE FAISS VECTOR DATABASE
        # ----------------------------------------------------

        with st.spinner(
            "Creating Ollama embeddings and FAISS database..."
        ):

            st.session_state.vectors = (
                FAISS.from_documents(
                    st.session_state.final_documents,
                    st.session_state.embeddings
                )
            )


# ============================================================
# DOCUMENT EMBEDDING BUTTON
# ============================================================

if st.button("Create Document Embeddings"):

    create_vector_embedding()

    st.success(
        " Vector Database is ready!"
    )


# ============================================================
# USER INPUT
# ============================================================

user_prompt = st.text_input(
    "Enter your question about the research papers:"
)


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

if user_prompt:

    # --------------------------------------------------------
    # CHECK VECTOR DATABASE
    # --------------------------------------------------------

    if "vectors" not in st.session_state:

        st.warning(
            "Please click 'Create Document Embeddings' "
            "before asking a question."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE DOCUMENT CHAIN
    # --------------------------------------------------------

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # --------------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = (
        st.session_state.vectors.as_retriever(
            search_kwargs={"k": 4}
        )
    )

    # --------------------------------------------------------
    # CREATE RETRIEVAL CHAIN
    # --------------------------------------------------------

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start = time.time()

    # --------------------------------------------------------
    # SEND QUESTION TO RAG CHAIN
    # --------------------------------------------------------

    response = retrieval_chain.invoke(
        {
            "input": user_prompt
        }
    )

    # --------------------------------------------------------
    # END TIMER
    # --------------------------------------------------------

    response_time = time.time() - start

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader(" Answer")

    st.write(
        response["answer"]
    )

    st.caption(
        f"Response time: {response_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # DISPLAY RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    with st.expander(
        " Document Similarity Search"
    ):

        for i, doc in enumerate(
            response["context"]
        ):

            st.write(
                f"### Document {i + 1}"
            )

            st.write(
                doc.page_content
            )

            # Display source PDF
            if "source" in doc.metadata:

                st.caption(
                    f"Source: {doc.metadata['source']}"
                )

            st.write(
                "--------------------------------"
            )