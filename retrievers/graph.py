from typing import List, Dict, Any
from langchain.schema import Document
import spacy
from neo4j import GraphDatabase
import json
import logging
import os

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        spacy_model: str = "en_core_web_sm"
    ):
        """Initialize the Neo4j graph retriever."""
        # Get credentials from environment if not provided
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not self.uri:
            raise ValueError("Neo4j URI must be provided either directly or through NEO4J_URI environment variable")
        if not self.password:
            raise ValueError("Neo4j password must be provided either directly or through NEO4J_PASSWORD environment variable")
            
        # Ensure URI has proper scheme
        if not self.uri.startswith(("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")):
            self.uri = f"bolt://{self.uri}"
            
        logger.info(f"Connecting to Neo4j at {self.uri}")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
            
        self.nlp = spacy.load(spacy_model)
        
        # Initialize the database schema
        self._init_schema()
    
    def _init_schema(self):
        """Initialize the Neo4j database schema with constraints."""
        try:
            with self.driver.session() as session:
                # Create constraints for unique nodes
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
            logger.info("Successfully initialized Neo4j schema")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {
            "PERSON": [],
            "ORG": []  # Organizations will be treated as Companies
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def add_document(self, doc: Document):
        """Add a document and its entities to the graph.
        
        Args:
            doc (Document): Document to add
        """
        try:
            entities = self.extract_entities(doc.page_content)
            
            with self.driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content
                    """,
                    doc_id=doc.metadata.get("id", str(hash(doc.page_content))),
                    content=doc.page_content
                )
                
                # Create entity nodes and relationships
                for entity_type, values in entities.items():
                    for value in values:
                        if entity_type == "PERSON":
                            session.run(
                                """
                                MERGE (p:Person {name: $name})
                                WITH p
                                MATCH (d:Document {id: $doc_id})
                                MERGE (d)-[:MENTIONS]->(p)
                                """,
                                name=value,
                                doc_id=doc.metadata.get("id", str(hash(doc.page_content)))
                            )
                        elif entity_type == "ORG":
                            session.run(
                                """
                                MERGE (c:Company {name: $name})
                                WITH c
                                MATCH (d:Document {id: $doc_id})
                                MERGE (d)-[:MENTIONS]->(c)
                                """,
                                name=value,
                                doc_id=doc.metadata.get("id", str(hash(doc.page_content)))
                            )
                
                # Add WORKS_AT relationships if "CEO" is mentioned
                if "ceo" in doc.page_content.lower():
                    for person in entities.get("PERSON", []):
                        for org in entities.get("ORG", []):
                            session.run(
                                """
                                MATCH (p:Person {name: $person})
                                MATCH (c:Company {name: $company})
                                MERGE (p)-[:WORKS_AT]->(c)
                                """,
                                person=person,
                                company=org
                            )
            logger.info(f"Successfully added document with ID {doc.metadata.get('id')}")
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents based on entity relationships.
        
        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            entities = self.extract_entities(query)
            
            with self.driver.session() as session:
                # Build the Cypher query based on extracted entities
                cypher_query = """
                MATCH (d:Document)
                WHERE """
                
                conditions = []
                params = {}
                
                for entity_type, values in entities.items():
                    for i, value in enumerate(values):
                        if entity_type == "PERSON":
                            conditions.append(f"EXISTS((d)-[:MENTIONS]->(:Person {{name: $person{i}}}))")
                            params[f"person{i}"] = value
                        elif entity_type == "ORG":
                            conditions.append(f"EXISTS((d)-[:MENTIONS]->(:Company {{name: $org{i}}}))")
                            params[f"org{i}"] = value
                
                if not conditions:
                    # If no entities found, try full-text search
                    cypher_query = """
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $search_text
                    RETURN d LIMIT $k
                    """
                    params = {"search_text": query, "k": k}
                else:
                    cypher_query += " OR ".join(conditions)
                    cypher_query += f" RETURN d LIMIT {k}"
                
                results = session.run(cypher_query, **params)
                documents = []
                
                for record in results:
                    doc = Document(
                        page_content=record["d"]["content"],
                        metadata={"retrieval_source": "graph"}
                    )
                    documents.append(doc)
                
                return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []
    
    def close(self):
        """Close the Neo4j driver connection."""
        try:
            if hasattr(self, 'driver'):
                self.driver.close()
                logger.info("Successfully closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}") 