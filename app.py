import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq

from src.helper import (
    get_pdf_text,
    split_text,
    create_embeddings,
    create_chain
)

def main():
    load_dotenv()

    st.set_page_config(page_title="Conversational RAG With PDF Uploads", layout="wide")
    st.title("Conversational RAG With PDF Uploads and Chat History")
    st.write("Upload PDF files and chat with their content.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Groq API key:", type="password")
        session_id = st.text_input("Session ID", value="default_session")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if not api_key:
        st.warning("Please enter the Groq API Key.")
        return

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session storage
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if uploaded_files:
        # Process PDFs
        docs = get_pdf_text(uploaded_files)
        splits = split_text(docs)
        vectorstore = create_embeddings(splits)

        # Build RAG chain
        rag_chain = create_chain(llm, vectorstore)

        # Session history getter
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key=None  # Because RAG returns a message object, not a dict
        )

        # UI
        st.subheader("Ask Questions")
        user_input = st.text_input("Your question:")

        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # response is an AIMessage object, so show response.content
            st.write("Assistant:", response.content)
            st.write("Chat History:", session_history.messages)

if __name__ == "__main__":
    main()
