from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
import time
import os


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)


class Chatbot:
    def __init__(self):
        GROQ_API = os.environ.get("GROQ_API")
        TAVILY_API = os.environ.get("TAVILY_API")

        CONNECTION_STRING = os.environ.get("DATABASE_URL")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.stream_handler = StreamHandler()

        self.llm = ChatGroq(
            model="llama3-8B-8192",
            streaming=True,
            callbacks=[self.stream_handler],
            max_tokens=512
        )

        self.vectorstore = PGVector(
            collection_name="vectordb",
            connection_string=CONNECTION_STRING,
            embedding_function=self.embeddings
        )

        self.original_similarity_search = self.vectorstore.similarity_search
        self.original_web_search = TavilySearchResults.invoke

        @tool
        def search_constitution_db(query: str) -> str:
            """Search the Indian Constitution database for relevant answers."""
            start = time.time()
            docs = self.original_similarity_search(query, k=2)
            end = time.time()
            self.tool_times["Vector DB Search"] = end - start
            return "Here is what I found in the Constitution:\n\n" + "\n\n".join(doc.page_content for doc in docs)

        @tool
        def search_govt_websites(query: str) -> str:
            """Search Indian government websites for recent constitutional updates."""
            start = time.time()
            restricted_query = f"{query} site:gov.in OR site:nic.in"
            search = TavilySearchResults(num_results=3)
            result = self.original_web_search(search, restricted_query)
            end = time.time()
            self.tool_times["Web Search"] = end - start
            return result

        tools = [search_constitution_db, search_govt_websites]

        prompt = hub.pull("hwchase17/react-chat")
        prompt.template = prompt.template.replace("{chat_history}", "{history}")
        prompt.input_variables = [v.replace("chat_history", "history") for v in prompt.input_variables]
        agent = create_react_agent(self.llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=20,
        )

        self.store = {}
        self.agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def invoke(self, query: str, session_id: str) -> dict:
        config = {"configurable": {"session_id": session_id}}
        self.tool_times = {}
        self.stream_handler.tokens = []

        start = time.time()
        try:
            response = self.agent_with_history.invoke({"input": query}, config=config)
            raw_output = response.get("output", "")
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            raw_output = ""

        end = time.time()
        self.tool_times["Answer Formation"] = end - start

        # Extract only the Final Answer
        if "Final Answer:" in raw_output:
            answer = raw_output.split("Final Answer:")[-1].strip()
        else:
            answer = raw_output.strip()

        return {
            "output": answer if answer else "I'm sorry, I couldn't generate a final answer.",
            "times": self.tool_times,
        }
