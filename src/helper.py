import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------
def get_pdf_text(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

# ---------------------------------------------------
# 2. Split text
# ---------------------------------------------------
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_documents(documents)

# ---------------------------------------------------
# 3. Create vector embeddings
# ---------------------------------------------------
def create_embeddings(splits):
    return FAISS.from_documents(documents=splits, embedding=embeddings)

# ---------------------------------------------------
# 4. Create retrieval QA chain (NEW LangChain)
# ---------------------------------------------------
def create_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()

    # --- Rewriting question based on chat history ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user question into a standalone question. "
         "Use chat history only for understanding, do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def contextualize_question(inputs):
        rewritten = llm.invoke(
            contextualize_q_prompt.format_messages(
                chat_history=inputs["chat_history"],
                input=inputs["input"]
            )
        )
        return rewritten.content

    # --- Retrieval step ---
    def retrieve_docs(inputs):
        standalone_q = contextualize_question(inputs)
        docs = retriever.invoke(standalone_q)
        return {"docs": docs, "question": standalone_q}

    # --- QA prompt ---
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering using retrieved context. "
         "If you don’t know the answer, say so. Answer in max three sentences.\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    def run_qa(inputs):
        context_text = "\n\n".join([doc.page_content for doc in inputs["docs"]])
        messages = qa_prompt.format_messages(
            context=context_text,
            chat_history=inputs["chat_history"],
            question=inputs["question"]
        )
        return llm.invoke(messages)

    # --- Build final RAG chain ---
    rag_chain = (
        RunnablePassthrough.assign(step=retrieve_docs)
        | run_qa
    )

    return rag_chain
