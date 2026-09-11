from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import time
import os

load_dotenv()


class StreamHandler(BaseCallbackHandler):
    """Collects tokens as the LLM streams them, so the UI can animate the response."""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)


class Chatbot:
    def __init__(self):
        GROQ_API = os.environ.get("GROQ_API")
        TAVILY_API = os.environ.get("TAVILY_API")
        CONNECTION_STRING = os.environ.get("DATABASE_URL")

        if not GROQ_API or not TAVILY_API or not CONNECTION_STRING:
            raise ValueError(
                "Missing one or more required environment variables: "
                "GROQ_API, TAVILY_API, DATABASE_URL. Check your .env file."
            )

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.stream_handler = StreamHandler()
        self.tool_times = {}  # initialized here too, not just in invoke(), so tools never hit a missing attribute

        # llama3-8b-8192 was deprecated by Groq (May 2025). Its suggested replacement,
        # llama-3.1-8b-instant, has since moved to Enterprise-only access (Contact Sales
        # pricing) — not available on a free-tier key. openai/gpt-oss-20b is currently
        # on the free/developer tier with real per-token pricing and normal rate limits.
        #
        # streaming=False here: combining streaming with native tool-calling causes
        # langchain-groq to leak a raw Stream object where a parsed message is expected
        # ("'Stream' object has no attribute 'dict'"). This doesn't affect the visible
        # typing animation — app.py reveals the finished answer character-by-character
        # itself, it isn't driven by token-level streaming from the LLM.
        #
        # gpt-oss models spend part of max_tokens on internal "reasoning" before writing
        # the final answer — on harder queries this can consume the whole budget, leaving
        # an empty final answer even though the run "finishes" successfully. reasoning_effort
        # keeps that shorter, and max_tokens is raised to leave headroom either way.
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=GROQ_API,
            streaming=False,
            max_tokens=1536,
            model_kwargs={"reasoning_effort": "low"}
        )

        try:
            self.vectorstore = PGVector(
                collection_name="vectordb",
                connection_string=CONNECTION_STRING,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the vector database. Check that Postgres is running "
                f"and DATABASE_URL is correct. Original error: {e}"
            ) from e

        # Created once instead of on every search_govt_websites call.
        # max_results is the correct field name (num_results is silently ignored by pydantic).
        # api_wrapper is built explicitly so it uses TAVILY_API instead of relying on
        # Tavily's own lookup, which expects an env var literally named TAVILY_API_KEY.
        self.web_search_tool = TavilySearchResults(
            max_results=3,
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=TAVILY_API)
        )

        @tool
        def search_constitution_db(query: str) -> str:
            """Search the Indian Constitution database for relevant answers."""
            start = time.time()
            docs = self.vectorstore.similarity_search(query, k=4)
            end = time.time()
            self.tool_times["Vector DB Search"] = end - start
            return "Here is what I found in the Constitution:\n\n" + "\n\n".join(doc.page_content for doc in docs)

        @tool
        def search_govt_websites(query: str) -> str:
            """Search Indian government websites for recent constitutional updates."""
            start = time.time()
            restricted_query = f"{query} site:gov.in OR site:nic.in"
            result = self.web_search_tool.invoke(restricted_query)
            end = time.time()
            self.tool_times["Web Search"] = end - start
            return result

        tools = [search_constitution_db, search_govt_websites]

        # create_tool_calling_agent (not the older create_react_agent) — gpt-oss-20b is a
        # native tool-calling model, and the older text-parsing ReAct agent conflicts with
        # that ("Tool choice is none, but model called a tool"). This also means we no
        # longer need to pull a prompt from LangChain Hub over the network at startup.
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that answers questions about the Indian "
             "Constitution. Below is context retrieved from the Constitution's actual text "
             "for the current question — treat it as your primary source of truth for "
             "factual questions, even if it conflicts with what you already believe. Do "
             "NOT state a specific count, number, or total (e.g. 'there are N of X') unless "
             "the retrieved context explicitly states that number — if it doesn't, say the "
             "retrieved context doesn't give a clear count rather than guessing one. If the "
             "retrieved context doesn't clearly answer the question, you may call "
             "search_constitution_db again with a refined query, or say you're not sure "
             "rather than guessing. Use the search_govt_websites tool only for recent "
             "updates, current events, or anything not covered by the Constitution's text "
             "itself.\n\nRetrieved context:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            # Bumped from 20s: a single query can involve an Ollama embedding call, a Postgres
            # vector search, a possible Tavily web search, and multiple Groq LLM calls in the
            # tool-calling loop — 20s was tight enough to cut off valid answers.
            max_execution_time=45,
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

        # Always retrieve context up front instead of leaving it to the model's
        # discretion — gpt-oss-20b will sometimes skip calling search_constitution_db
        # entirely and answer from its own (unreliable) memory otherwise, regardless of
        # what the system prompt instructs. The tool stays available too, in case the
        # agent wants to run a second, more targeted search.
        start_context = time.time()
        try:
            context_docs = self.vectorstore.similarity_search(query, k=4)
            context = "\n\n".join(doc.page_content for doc in context_docs)
        except Exception as e:
            print(f"[WARN] Context pre-fetch failed: {e}")
            context = ""
        self.tool_times["Context Retrieval"] = time.time() - start_context

        start = time.time()
        error_message = None
        try:
            response = self.agent_with_history.invoke(
                {"input": query, "context": context}, config=config
            )
            raw_output = response.get("output", "")
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            error_message = str(e)
            raw_output = ""

        end = time.time()
        self.tool_times["Answer Formation"] = end - start

        # Extract only the Final Answer
        if "Final Answer:" in raw_output:
            answer = raw_output.split("Final Answer:")[-1].strip()
        else:
            answer = raw_output.strip()

        if not answer:
            answer = "I'm sorry, I couldn't generate a final answer."
            if error_message:
                # Printed to the console only — app.py formats every value in `times` as a
                # float (seconds), so a string here would break that display.
                print(f"[ERROR] Final answer was empty. Underlying error: {error_message}")

        return {
            "output": answer,
            "times": self.tool_times,
        }