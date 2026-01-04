from typing import List
from langchain.schema import Document
import duckdb
import logging

logger = logging.getLogger(__name__)

class SQLRetriever:
    def __init__(self, db_path: str):
        """Initialize the SQL retriever with DuckDB connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents based on SQL query.
        
        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            # Execute the query
            result = self.conn.execute(query).fetchdf()
            
            # Convert results to documents
            documents = []
            for _, row in result.iterrows():
                # Convert row to string representation
                content = " | ".join(f"{col}: {val}" for col, val in row.items())
                doc = Document(
                    page_content=content,
                    metadata={
                        "retrieval_source": "sql",
                        "query": query
                    }
                )
                documents.append(doc)
            
            return documents[:k]
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return []
    
    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close() 