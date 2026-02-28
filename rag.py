from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def load_and_chunk(pdf_path):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Chunks created: {len(chunks)}")
    return chunks

def embed_and_store(chunks):
    print("Loading embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"Stored {vectorstore._collection.count()} chunks.")
    return vectorstore

def build_qa_chain(vectorstore):
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

def ask(chain, question):
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")

def main():
    print("=================================")
    print("   📄 Document Q&A with Groq     ")
    print("=================================")

    chunks = load_and_chunk("data/Rebecca-Shirievo-resume.pdf")
    vectorstore = embed_and_store(chunks)
    chain = build_qa_chain(vectorstore)

    print("\nDocument loaded. Ask me anything about it.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break

        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")

main()