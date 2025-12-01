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

    # Rewriting prompt
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user question into a standalone question. "
         "Use chat history for understanding but do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # 1️⃣ Rewrite Question
    def rewrite_step(inputs):
        # RunnableWithMessageHistory injects chat history into inputs["chat_history"]
        rewritten = llm.invoke(
            rewrite_prompt.format_messages(
                input=inputs["input"],
                chat_history=inputs["chat_history"]
            )
        )
        return {
            "rewritten_question": rewritten.content,
            "chat_history": inputs["chat_history"]
        }

    # 2️⃣ Retrieve documents
    def retrieve_step(inputs):
        docs = retriever.invoke(inputs["rewritten_question"])
        return {
            "docs": docs,
            "question": inputs["rewritten_question"],
            "chat_history": inputs["chat_history"]
        }

    # QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for QA tasks based on retrieved context. "
         "If unsure, say 'I don't know'. Max three sentences.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    # 3️⃣ Answer step
    def answer_step(inputs):
        context_text = "\n\n".join([doc.page_content for doc in inputs["docs"]])

        messages = qa_prompt.format_messages(
            context=context_text,
            question=inputs["question"],
            chat_history=inputs["chat_history"]
        )

        return llm.invoke(messages)

    # Build final chain
    return (
        RunnablePassthrough()
        | rewrite_step
        | retrieve_step
        | answer_step
    )
