import streamlit as st
import uuid
from chatbot_local import Chatbot
import time
import streamlit.components.v1 as components
import html


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
if st.sidebar.button("+ New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {"role": "assistant", "content": "New chat started. Ask me anything!"}
    ]
    st.rerun()

# Derive session label from last question
if st.session_state.messages:
    last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
    if last_user_msg:
        short_label = " ".join(last_user_msg.strip().split()[:3]) + "..."
    else:
        short_label = "New session"
else:
    short_label = "New session"

st.sidebar.info(f"🆔 Chat: `{short_label}`")

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
                result = chatbot.invoke(prompt, st.session_state.session_id)

                response_text = result["output"]
                timings = result.get("times", {})

                # Live stream response
                placeholders = st.empty()
                display_text = ""
                for char in response_text:
                    display_text += char
                    placeholders.markdown(display_text + "▌")
                    time.sleep(0.01)
                placeholders.markdown(display_text)

                sanitized = html.escape(display_text).replace("`", "\\`")  # prevent JS break

                components.html(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-top: 10px; margin-bottom: 10px;">
                        <button id="copy-btn"
                            style="
                                background-color: #333;
                                color: white;
                                border: 1px solid #666;
                                border-radius: 6px;
                                padding: 6px 12px;
                                cursor: pointer;
                                font-size: 18px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3);"
                            title="Copy to clipboard">📋</button>
                    </div>
                
                    <script>
                        const copyButton = document.getElementById("copy-btn");
                        copyButton.onclick = () => {{
                            navigator.clipboard.writeText(`{sanitized}`).then(() => {{
                                const toast = document.createElement('div');
                                toast.innerText = "✅ Copied!";
                                toast.style.position = 'fixed';
                                toast.style.bottom = '30px';
                                toast.style.right = '30px';
                                toast.style.padding = '10px 20px';
                                toast.style.backgroundColor = '#4CAF50';
                                toast.style.color = 'white';
                                toast.style.borderRadius = '5px';
                                toast.style.boxShadow = '0 2px 6px rgba(0,0,0,0.3)';
                                toast.style.zIndex = 10000;
                                document.body.appendChild(toast);
                                setTimeout(() => document.body.removeChild(toast), 2000);
                            }});
                        }}
                    </script>
                    """,
                    height=70,
                )
                

                # Save message
                st.session_state.messages.append({"role": "assistant", "content": display_text})

                # Show time details
                with st.expander("⏱ Time Taken Details"):
                    for step, t in timings.items():
                        st.write(f"**{step}**: {t:.4f} seconds")


            except Exception as e:
                error_message = f"Sorry, I encountered an error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})