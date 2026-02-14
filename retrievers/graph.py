from typing import List, Dict, Any
from langchain_core.documents import Document
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
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
                # Basketball schema
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Coach) REQUIRE c.name IS UNIQUE")
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
    
    def _tokenize(self, query: str) -> List[str]:
        """Build search tokens from query (full phrase, no spaces, and words > 2 chars)."""
        q = query.strip().lower()
        return [q, q.replace(" ", "")] + [w for w in q.split() if len(w) > 2]

    def _run_entity_queries(self, session, tokens: List[str]) -> List[str]:
        """Run token-driven entity lookups (Team, Player, Coach) from a single config."""
        # One pattern: match by tokens, optional relationships, return structured row. Formatter turns row into text.
        entity_config = [
            (
                """
                MATCH (t:Team)
                WHERE any(n IN $tokens WHERE toLower(t.name) CONTAINS n OR n CONTAINS toLower(t.name))
                OPTIONAL MATCH (c:Coach)-[:COACHES]->(t)
                OPTIONAL MATCH (p:Player)-[r:PLAYS_FOR]->(t)
                RETURN t.name AS team, t.city AS city, c.name AS coach,
                       collect(DISTINCT {name: p.name, ppg: r.ppg, career_high: r.career_high}) AS players
                """,
                lambda rec: self._format_team_row(rec),
            ),
            (
                """
                MATCH (p:Player)-[r:PLAYS_FOR]->(t:Team)
                WHERE any(n IN $tokens WHERE toLower(p.name) CONTAINS n OR n CONTAINS toLower(p.name))
                OPTIONAL MATCH (p)-[:TEAMMATE_OF]->(m:Player)
                RETURN p.name AS player, t.name AS team, r.ppg AS ppg, r.career_high AS career_high,
                       collect(DISTINCT m.name) AS teammates
                """,
                lambda rec: self._format_player_row(rec),
            ),
            (
                """
                MATCH (c:Coach)-[:COACHES]->(t:Team)
                WHERE any(n IN $tokens WHERE toLower(c.name) CONTAINS n OR n CONTAINS toLower(c.name))
                OPTIONAL MATCH (p:Player)-[r:PLAYS_FOR]->(t)
                RETURN c.name AS coach, t.name AS team, collect(DISTINCT {name: p.name, ppg: r.ppg}) AS players
                """,
                lambda rec: self._format_coach_row(rec),
            ),
        ]
        facts = []
        for cypher, formatter in entity_config:
            for rec in session.run(cypher, tokens=tokens):
                line = formatter(rec)
                if line:
                    facts.append(line)
        return facts

    def _format_team_row(self, rec: Dict[str, Any]) -> str:
        team, city, coach, players = rec.get("team"), rec.get("city"), rec.get("coach"), rec.get("players") or []
        if not team:
            return ""
        parts = [f"Team: {team}" + (f" ({city})" if city else "")]
        if coach:
            parts.append(f"Coach: {coach}")
        for x in players:
            if x.get("name"):
                parts.append(f"Player: {x['name']}" + (f", PPG: {x.get('ppg')}, Career high: {x.get('career_high')}" if x.get("ppg") is not None else ""))
        return " | ".join(parts) if len(parts) > 1 else ""

    def _format_player_row(self, rec: Dict[str, Any]) -> str:
        player = rec.get("player")
        if not player:
            return ""
        team, ppg, ch = rec.get("team"), rec.get("ppg"), rec.get("career_high")
        mates = [m for m in (rec.get("teammates") or []) if m]
        parts = [f"Player: {player}", f"Team: {team}"]
        if ppg is not None:
            parts.append(f"PPG: {ppg}")
        if ch is not None:
            parts.append(f"Career high: {ch}")
        if mates:
            parts.append(f"Teammates: {', '.join(mates)}")
        return " | ".join(parts)

    def _format_coach_row(self, rec: Dict[str, Any]) -> str:
        coach, team = rec.get("coach"), rec.get("team")
        if not coach:
            return ""
        players = rec.get("players") or []
        parts = [f"Coach: {coach}", f"Team: {team}"]
        for x in players:
            if x.get("name"):
                parts.append(f"Player: {x['name']}" + (f" (PPG: {x.get('ppg')})" if x.get("ppg") is not None else ""))
        return " | ".join(parts)

    def _run_relationship_queries(self, session, query_lower: str, tokens: List[str]) -> List[str]:
        """Run relationship lookups when query contains trigger words; all use tokens to scope."""
        # (trigger_substrings, cypher, row formatter)
        rel_config = [
            (
                ["teammate", "teammates", "plays with"],
                """
                MATCH (a:Player)-[:TEAMMATE_OF]->(b:Player)
                WHERE any(n IN $tokens WHERE toLower(a.name) CONTAINS n OR n CONTAINS toLower(a.name))
                RETURN a.name AS a, b.name AS b
                """,
                lambda rec: f"Teammates: {rec['a']} and {rec['b']}" if rec.get("a") and rec.get("b") else "",
            ),
        ]
        facts = []
        for triggers, cypher, formatter in rel_config:
            if not any(t in query_lower for t in triggers):
                continue
            for rec in session.run(cypher, tokens=tokens):
                line = formatter(rec)
                if line:
                    facts.append(line)
        return facts

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve from basketball graph using tokenized query; entity and relationship lookups are config-driven."""
        try:
            q_lower = query.strip().lower()
            tokens = self._tokenize(query)
            facts = []
            with self.driver.session() as session:
                facts.extend(self._run_entity_queries(session, tokens))
                facts.extend(self._run_relationship_queries(session, q_lower, tokens))

            seen = set()
            unique = []
            for f in facts:
                if f and f not in seen:
                    seen.add(f)
                    unique.append(f)
            if not unique:
                unique = ["Graph has teams (Celtics, Lakers, Warriors), players, and coaches. Ask by name, e.g. 'Who coaches the Lakers?' or 'Jayson Tatum stats'."]
            return [
                Document(page_content=text, metadata={"retrieval_source": "graph"})
                for text in unique[:k]
            ]
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