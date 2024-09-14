from info_retreival import create_rag_chain, invoke_rag_chain
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st


# Initialize chat history and context
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import os
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

loaded_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = loaded_vector_store.as_retriever()

rag_chain=create_rag_chain(retriever, llm)


# Sidebar for adjustable parameters
st.sidebar.header("Model Settings")

# Temperature slider (0.0 to 1.0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)  # Default to 0.7

# Top_P slider (0.0 to 1.0)
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 0.9)  # Default to 0.9

# Frequency Penalty input (0.0 to 2.0)
frequency_penalty = st.sidebar.number_input("Frequency Penalty", 0.0, 2.0, 0.0)  # Default to 0.0

# Presence Penalty input (0.0 to 2.0)
presence_penalty = st.sidebar.number_input("Presence Penalty", 0.0, 2.0, 0.0)  # Default to 0.0

# Max Tokens input (100 to 1024)
max_tokens = st.sidebar.number_input("Max Tokens", 100, 1024, 512)


prompt = st.chat_input("What's up?", key="main_input")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # user_input="tell me about australia's real estate scene?"
    response=invoke_rag_chain(prompt, rag_chain, temperature=temperature, max_t=max_tokens, top_p=top_p, fp=frequency_penalty, pp=presence_penalty)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
