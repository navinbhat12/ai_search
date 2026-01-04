from typing import List, Dict, Any, Optional
from langchain.schema import Document
from retrievers.vector import QdrantRetriever
from retrievers.sql import DuckDBRetriever

class HybridRetriever:
    def __init__(
        self,
        vector_retriever: QdrantRetriever,
        sql_retriever: DuckDBRetriever,
        default_answer: str = "I couldn't find a relevant answer to your question."
    ):
        """Initialize the hybrid retriever.
        
        Args:
            vector_retriever (QdrantRetriever): Vector-based retriever
            sql_retriever (DuckDBRetriever): SQL-based retriever
            default_answer (str): Default answer when no results are found
        """
        self.vector_retriever = vector_retriever
        self.sql_retriever = sql_retriever
        self.default_answer = default_answer
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using hybrid approach with fallback.
        
        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        # Try vector retrieval first
        vector_results = self.vector_retriever.retrieve(query, k=k)
        
        if vector_results:
            return vector_results
        
        # If vector retrieval fails, try SQL retrieval
        sql_results = self.sql_retriever.retrieve(query, k=k)
        
        if sql_results:
            return sql_results
        
        # If both retrievers fail, return default answer
        return [Document(page_content=self.default_answer, metadata={})]
    
    def close(self):
        """Close all retriever connections."""
        self.vector_retriever.close()
        self.sql_retriever.close() 