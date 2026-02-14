from typing import List, Dict, Any, Union, Optional
from langchain_core.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish
from agents.hf_llm import get_hf_chat_llm
from langchain_core.documents import Document
from langchain_classic.chains import LLMChain
from langchain_classic.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain_core.prompts import StringPromptTemplate
import re
import logging
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from retrievers.graph import GraphRetriever
from retrievers.sql import SQLRetriever
from retrievers.vector import VectorRetriever

# Load environment variables
load_dotenv()

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('router_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RouterPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the router agent."""
    
    template: str = """You are an intelligent query router that determines which retrieval system to use.
    
Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important: After you receive an Observation from a tool, you must respond with "Thought: I now know the final answer" and then "Final Answer: ...". Do not call the same tool again. Your Final Answer must ONLY use information that appears in the Observation above. You may reword it into a clear, readable sentence or short paragraph and omit details that are not relevant to the question—but you must NOT add any facts, names, or information that are not in the Observation. If the data does not say something, do not say it.

Question: {input}
{agent_scratchpad}"""
    
    def __init__(self, tools, **kwargs):
        # Remove tools from kwargs before passing to super() to avoid Pydantic validation
        tools_list = tools
        # Pre-compute tools strings before initialization
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools_list])
        tool_names_str = ", ".join([tool.name for tool in tools_list])
        # Initialize parent first
        super().__init__(**kwargs)
        # Then set private attributes using object.__setattr__ to bypass Pydantic
        object.__setattr__(self, "_tools_str", tools_str)
        object.__setattr__(self, "_tool_names_str", tool_names_str)

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (handle case where it might not exist yet)
        intermediate_steps = kwargs.pop("agent_scratchpad", [])
        thoughts = ""
        if intermediate_steps:
            for action, observation in intermediate_steps:
                thoughts += f"\nThought: I need to use {action.tool} with input {action.tool_input}\nAction: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        kwargs["agent_scratchpad"] = thoughts
        # Ensure tools and tool_names are included
        if "tools" not in kwargs:
            kwargs["tools"] = self._tools_str
        if "tool_names" not in kwargs:
            kwargs["tool_names"] = self._tool_names_str
        return self.template.format(**kwargs)

class RouterOutputParser(AgentOutputParser):
    """Custom output parser for the router agent."""
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """Parse the LLM output into an AgentAction or AgentFinish."""
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
        
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": "I could not determine the next action."},
                log=llm_output
            )
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

class RouterAgent:
    def __init__(
        self,
        sql_retriever: SQLRetriever,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        huggingfacehub_api_token: str
    ):
        """Initialize the router agent with all retrievers."""
        self.sql_retriever = sql_retriever
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        
        # Initialize tools with more specific descriptions
        self.tools = [
            Tool(
                name="SQLRetriever",
                func=self._sql_retrieve,
                description="""Use this for structured data questions. Pass the user's question in natural language (e.g. "Which players average over 25 points per game?").
Good for: basketball stats (teams, players, games), aggregations, filters, rankings, counts, and any question about tabular data. Do not write SQL yourself—pass the question as-is."""
            ),
            Tool(
                name="VectorRetriever",
                func=self._vector_retrieve,
                description="""Use this for:
- Semantic search queries
- Finding similar content
- Understanding product features
- General information retrieval
- When you need to understand the meaning of text"""
            ),
            Tool(
                name="GraphRetriever",
                func=self._graph_retrieve,
                description="""Use this for:
- Company information and relationships
- CEO and leadership queries
- Company acquisitions and partnerships
- Entity relationships
- When you need to traverse a knowledge graph"""
            )
        ]
        
        # LLM via Hugging Face OpenAI-compatible API (router.huggingface.co)
        self.llm = get_hf_chat_llm(
            api_key=huggingfacehub_api_token,
            temperature=0.1,
            max_tokens=512,
        )
        
        # Initialize output parser
        self.output_parser = RouterOutputParser()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize trace log and last observation (for response synthesis)
        self.trace_log = []
        self._last_observation: Optional[str] = None
        self._last_intermediate_steps: List[tuple] = []  # (AgentAction, observation) for UI pipeline
        
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with routing logic."""
        prompt = RouterPromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            tools=self.tools
        )
        
        # Create LLM chain
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        # Create agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            return_intermediate_steps=True,
        )
    
    def _log_trace(self, event_type: str, data: Dict[str, Any]):
        """Log a trace event with timestamp."""
        trace_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.trace_log.append(trace_event)
        logger.info(f"Trace: {json.dumps(trace_event, indent=2)}")
    
    def _sql_retrieve(self, query: str) -> str:
        """Execute SQL retrieval and log the operation."""
        self._log_trace("sql_retrieval_start", {"query": query})
        try:
            results = self.sql_retriever.retrieve(query)
            formatted_results = self._format_results(results, "SQL")
            self._last_observation = formatted_results
            self._log_trace("sql_retrieval_complete", {
                "query": query,
                "result_count": len(results),
                "retrieved_content": formatted_results,
            })
            return formatted_results
        except Exception as e:
            self._log_trace("sql_retrieval_error", {
                "query": query,
                "error": str(e)
            })
            raise
    
    def _vector_retrieve(self, query: str) -> str:
        """Execute vector retrieval and log the operation."""
        self._log_trace("vector_retrieval_start", {"query": query})
        try:
            results = self.vector_retriever.retrieve(query)
            formatted_results = self._format_results(results, "Vector")
            self._last_observation = formatted_results
            self._log_trace("vector_retrieval_complete", {
                "query": query,
                "result_count": len(results),
                "retrieved_content": formatted_results,
            })
            return formatted_results
        except Exception as e:
            self._log_trace("vector_retrieval_error", {
                "query": query,
                "error": str(e)
            })
            raise
    
    def _graph_retrieve(self, query: str) -> str:
        """Execute graph retrieval and log the operation."""
        self._log_trace("graph_retrieval_start", {"query": query})
        try:
            results = self.graph_retriever.retrieve(query)
            formatted_results = self._format_results(results, "Graph")
            self._last_observation = formatted_results
            self._log_trace("graph_retrieval_complete", {
                "query": query,
                "result_count": len(results),
                "retrieved_content": formatted_results,
            })
            return formatted_results
        except Exception as e:
            self._log_trace("graph_retrieval_error", {
                "query": query,
                "error": str(e)
            })
            raise
    
    def _format_results(self, results: List[Document], source: str) -> str:
        """Format retrieval results into a string."""
        if not results:
            return f"No results found using {source} retriever."
        
        formatted = f"Found {len(results)} results from {source} retriever:\n"
        for i, doc in enumerate(results, 1):
            formatted += f"\n{i}. {doc.page_content}"
            if doc.metadata:
                formatted += f"\n   Metadata: {doc.metadata}"
        return formatted

    def _synthesize_response(self, query: str, observation: str) -> str:
        """Format the tool observation into a readable answer using only the provided data."""
        from langchain_core.messages import HumanMessage
        prompt = f"""The user asked: "{query}"

Here is the data we found:
{observation}

Rewrite this into one or two clear, readable sentences that answer the question. You may reword and omit details that are not relevant to the question. You must NOT add any facts, names, or information that are not in the data above—only format what is there. If there are no results, say so in one sentence."""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = getattr(response, "content", str(response)).strip()
            return text if text else observation
        except Exception as e:
            logger.warning(f"Response synthesis failed: {e}")
            return observation
    
    def route_query(self, query: str) -> str:
        """Route a query to the appropriate retriever using the agent."""
        self._last_observation = None
        self._last_intermediate_steps = []
        self.trace_log = []  # Reset so this query's trace is self-contained
        self._log_trace("query_received", {"query": query})
        try:
            out = self.agent.invoke({"input": query, "agent_scratchpad": []})
            # invoke returns dict with "output" and "intermediate_steps" when return_intermediate_steps=True
            result = out.get("output", "") if isinstance(out, dict) else out
            self._last_intermediate_steps = out.get("intermediate_steps", []) if isinstance(out, dict) else []

            # If agent hit iteration limit without a Final Answer, synthesize one from last tool output
            if result and "Agent stopped due to iteration limit" in result and self._last_observation:
                synthesis_prompt = f'''The user asked: "{query}"

Here is the data we found:
{self._last_observation}

Rewrite this into one or two clear, readable sentences that answer the question. You may reword and omit details that are not relevant to the question. You must NOT add any facts, names, or information that are not in the data above—only format what is there. If there are no results, say so in one sentence.'''
                self._log_trace("prompt_augmented", {
                    "query": query,
                    "retrieved_info": self._last_observation,
                    "prompt_sent": synthesis_prompt,
                })
                result = self._synthesize_response(query, self._last_observation)
                self._log_trace("response_synthesized", {"query": query})
            self._log_trace("query_complete", {
                "query": query,
                "result": result,
            })
            self._log_trace("final_response", {"response": result})
            return result
        except Exception as e:
            self._log_trace("query_error", {
                "query": query,
                "error": str(e)
            })
            return f"Error processing query: {str(e)}"
    
    def get_trace_log(self) -> List[Dict[str, Any]]:
        """Get the complete trace log."""
        return self.trace_log
    
    def close(self):
        """Close all retriever connections."""
        self._log_trace("cleanup_start", {})
        self.sql_retriever.close()
        self.vector_retriever.close()
        self.graph_retriever.close()
        self._log_trace("cleanup_complete", {})


# Global router agent instance (singleton pattern)
_router_agent = None

def ask_router(query: str) -> str:
    """Main entry point for querying the router agent."""
    global _router_agent
    
    # Initialize router agent if not already initialized
    if _router_agent is None:
        try:
            logger.info("Initializing RouterAgent with all retrievers...")
            
            # Get credentials from environment
            huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not huggingface_token:
                return "Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables"
            
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if not qdrant_url:
                return "Error: QDRANT_URL not found in environment variables"
            
            # Initialize retrievers (SQL retriever uses same HF model for NL-to-SQL)
            sql_retriever = SQLRetriever(
                db_path=os.getenv("DUCKDB_PATH", "duckdb.db"),
                huggingfacehub_api_token=huggingface_token,
            )
            vector_retriever = VectorRetriever(
                collection_name="docs",
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key
            )
            graph_retriever = GraphRetriever()  # Uses environment variables automatically
            
            # Initialize router agent
            _router_agent = RouterAgent(
                sql_retriever=sql_retriever,
                vector_retriever=vector_retriever,
                graph_retriever=graph_retriever,
                huggingfacehub_api_token=huggingface_token
            )
            
            logger.info("RouterAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RouterAgent: {str(e)}")
            return f"Error initializing router agent: {str(e)}"
    
    # Route the query
    try:
        result = _router_agent.route_query(query)
        return result
    except Exception as e:
        logger.error(f"Error routing query: {str(e)}")
        return f"Error processing query: {str(e)}"


def get_router_trace_log() -> List[Dict[str, Any]]:
    """Return the trace log for the last route_query (for UI/debugging)."""
    global _router_agent
    if _router_agent is None:
        return []
    return _router_agent.get_trace_log()


def get_router_last_intermediate_steps() -> List[tuple]:
    """Return (AgentAction, observation) steps from the last route_query for pipeline display."""
    global _router_agent
    if _router_agent is None:
        return []
    return getattr(_router_agent, "_last_intermediate_steps", [])


def get_router_pipeline_steps() -> List[Dict[str, Any]]:
    """Return last run's agent steps as JSON-serializable dicts for UI: thought, action, action_input, observation."""
    steps = get_router_last_intermediate_steps()
    out = []
    for action, observation in steps:
        out.append({
            "thought": getattr(action, "log", str(action)),
            "action": getattr(action, "tool", ""),
            "action_input": getattr(action, "tool_input", ""),
            "observation": observation,
        })
    return out