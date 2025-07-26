import streamlit as st
import uuid
from chatbot import Chatbot


st.set_page_config(
    page_title="Constitutional AI Assistant",
    page_icon="🇮🇳",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Constitutional AI Assistant 🇮🇳")
st.caption("Guide to the Indian Constitution")

@st.cache_resource
def load_chatbot():
    return Chatbot()

try:
    chatbot = load_chatbot()
    st.sidebar.success("Chatbot loaded successfully!")
except Exception as e:
    st.error(f"failed to load chatbot: {e}")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with the Indian Constitution today?"}
    ]

st.sidebar.title("Session Control")
if st.sidebar.button("New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {"role": "assistant", "content": "New chat started. Ask me anything!"}
    ]
    st.rerun()

st.sidebar.info(f"Session ID: `{st.session_state.session_id}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the Indian Constitution..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                response = chatbot.invoke(prompt, st.session_state.session_id)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})