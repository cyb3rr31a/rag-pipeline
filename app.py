import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A")
st.caption("Upload a PDF and ask questions about it")

# --- Helper functions ---
def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(pages)

def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore

def build_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say "I don't have enough information to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# --- Session state ---
# Keeps the chain alive between interactions without re-embedding
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.chain is None:
    with st.spinner("Reading and indexing your document..."):
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Clear old ChromaDB if exists
        import shutil
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        chunks = load_and_chunk(temp_path)
        vectorstore = embed_and_store(chunks)
        st.session_state.chain = build_chain(vectorstore)
        os.remove(temp_path)

    st.success(f"Document indexed — {len(chunks)} chunks created. Ask away!")

# --- Chat interface ---
if st.session_state.chain:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Upload a PDF to get started")