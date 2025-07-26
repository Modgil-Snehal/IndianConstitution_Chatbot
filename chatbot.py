from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from typing import Dict
from datetime import datetime
import uuid
import time
import os


class Chatbot:
    def __init__(self):
        os.environ["GROQ_API_KEY"] = "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"
        os.environ["TAVILY_API_KEY"] = "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"
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBRyc9-6UqIxK4ugBgt2HywoXo9iZBz4QQ"

        CONNECTION_STRING = "postgresql://postgres:5107@localhost:5432/constdb"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGroq(model="llama3-8B-8192")
        self.vectorstore = PGVector(
            collection_name="vectordb",
            connection_string=CONNECTION_STRING,
            embedding_function=self.embeddings
        )
        
        @tool
        def search_constitution_db(query: str) -> str:
            """
            Use this tool to search the local database for the specific text of articles,
            amendments, and schedules of the Indian Constitution. This is the fastest and
            most reliable source for foundational constitutional text.
            """
            docs = self.vectorstore.similarity_search(query, k=5)
            return "\n\n".join(doc.page_content for doc in docs)
        
        @tool
        def search_govt_websites(query: str) -> str:
            """
            Use this tool to search official Indian government websites for recent news,
            press releases, modern legal interpretations, or updated amendments related
            to the Indian Constitution that might not be in the local database.
            """
            restricted_query = f"{query} site:gov.in OR site:nic.in"
            search = TavilySearchResults(num_results=3,)
            return search.invoke(restricted_query)

        tools = [search_constitution_db, search_govt_websites]

        prompt = hub.pull("hwchase17/react-chat")
        prompt.template = prompt.template.replace("{chat_history}", "{history}")
        prompt.input_variables = [v.replace("chat_history", "history") for v in prompt.input_variables]
        agent = create_react_agent(self.llm, tools, prompt) 
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
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
    
    def invoke(self, query: str, session_id: str) -> str:
        """
        The main method to interact with the chatbot.
        """
        config = {"configurable": {"session_id": session_id}}
        response = self.agent_with_history.invoke({"input": query}, config=config)
        return response.get("output", "I'm sorry, I encountered an issue and couldn't provide a response.")
