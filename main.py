from info_retreival import create_rag_chain
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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


loaded_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = loaded_vector_store.as_retriever()



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

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=temperature, 
    max_tokens=max_tokens, 
    top_p=top_p, 
    frequency_penalty=frequency_penalty, 
    presence_penalty=presence_penalty
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an AI trained on the same updated(up until now) information that accredited real estate agents in Queensland, Australia learn from. "
    "But don't state the timeline of your training. "
    "Your role is to provide clear and accurate information about the real estate process, whether users are buying, selling, renting, or seeking advice. "
    "Guide users through their inquiries and connect them with the appropriate real estate professionals when needed. "
    "When responding to questions, use the following pieces of retrieved context to craft your answer. "
    "If you don't have enough information, avoid simply stating that you don't know; instead, make an effort to provide the best possible answer using the available context. "
    "If a request falls outside your context or scope, offer advice where possible, but clearly indicate that it's beyond your expertise. "
    "Keep your responses concise, using a maximum of three sentences."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
prompt = st.chat_input("What's up?", key="main_input")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # user_input="tell me about australia's real estate scene?"
    response=conversational_rag_chain.invoke(
            {"input": prompt},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"]
    
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
