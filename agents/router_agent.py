from typing import List, Dict, Any, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import HuggingFaceHub
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.agents import AgentOutputParser
import re
import logging
import json
from datetime import datetime
from retrievers.graph import GraphRetriever
from retrievers.sql import SQLRetriever
from retrievers.vector import VectorRetriever

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
    
    template = """You are an intelligent query router that determines which retrieval system to use.
    
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

Question: {input}
{agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        # Get the intermediate steps
        intermediate_steps = kwargs.pop("agent_scratchpad")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        kwargs["agent_scratchpad"] = thoughts
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
                description="""Use this for:
- Customer order analysis
- Sales data queries
- Numerical aggregations
- Time-based queries (e.g., monthly, quarterly data)
- Statistical questions about orders, customers, or products"""
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
        
        # Initialize LLM with a free model from HuggingFace
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Free, open-source model
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=huggingfacehub_api_token
        )
        
        # Initialize output parser
        self.output_parser = RouterOutputParser()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize trace log
        self.trace_log = []
        
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
            max_iterations=3  # Limit iterations to prevent infinite loops
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
            self._log_trace("sql_retrieval_complete", {
                "query": query,
                "result_count": len(results)
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
            self._log_trace("vector_retrieval_complete", {
                "query": query,
                "result_count": len(results)
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
            self._log_trace("graph_retrieval_complete", {
                "query": query,
                "result_count": len(results)
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
    
    def route_query(self, query: str) -> str:
        """Route a query to the appropriate retriever using the agent."""
        self._log_trace("query_received", {"query": query})
        try:
            result = self.agent.run(query)
            self._log_trace("query_complete", {
                "query": query,
                "result": result
            })
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


def ask_router(query: str) -> str:
    return "RouterAgent is not configured yet. Please wire retrievers and credentials."