from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import duckdb
import logging
import re

logger = logging.getLogger(__name__)

# Forbidden SQL keywords (guardrail: SELECT only)
_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


class SQLRetriever:
    def __init__(self, db_path: str, huggingfacehub_api_token: Optional[str] = None):
        """Initialize the SQL retriever with DuckDB connection.
        If huggingfacehub_api_token is set, natural language questions will be
        converted to SQL using the HuggingFace model; otherwise only raw SQL is accepted.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._hf_token = huggingfacehub_api_token
        self._llm = None

    def _get_llm(self):
        """Lazy-init LLM for NL-to-SQL via Hugging Face OpenAI-compatible API."""
        if self._llm is not None:
            return self._llm
        if not self._hf_token:
            return None
        try:
            from agents.hf_llm import get_hf_chat_llm
            self._llm = get_hf_chat_llm(
                api_key=self._hf_token,
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning(f"Could not create HF LLM for NL-to-SQL: {e}")
        return self._llm

    def get_schema(self) -> str:
        """Return a string description of the database schema for the LLM."""
        try:
            df = self.conn.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main'
                ORDER BY table_name, ordinal_position
            """).fetchdf()
            if df.empty:
                return "No tables found."
            lines = []
            cur_table = None
            for _, row in df.iterrows():
                t, c, d = row["table_name"], row["column_name"], row["data_type"]
                if t != cur_table:
                    cur_table = t
                    lines.append(f"\nTable: {t}")
                lines.append(f"  - {c} ({d})")
            return "\n".join(lines).strip() or "No schema."
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return "Could not read schema."

    def _validate_select_only(self, sql: str) -> bool:
        """Return True if the statement is SELECT-only (or WITH ... SELECT)."""
        s = sql.strip()
        # Remove single-line and block comments
        s = re.sub(r"--[^\n]*", "", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = " ".join(s.split()).strip()
        if not s:
            return False
        upper = s.upper()
        # Must start with SELECT or WITH
        if upper.startswith("SELECT"):
            pass
        elif upper.startswith("WITH"):
            if "SELECT" not in upper:
                return False
        else:
            return False
        if _FORBIDDEN_SQL_KEYWORDS.search(s):
            return False
        return True

    def _nl_to_sql(self, question: str) -> str:
        """Convert natural language to a single SELECT using schema + LLM."""
        llm = self._get_llm()
        if not llm:
            logger.warning("No LLM available for NL-to-SQL; returning empty.")
            return ""

        schema = self.get_schema()
        prompt = PromptTemplate.from_template("""Database schema:
{schema}

Question: {question}

Respond with exactly one SQL SELECT statement for DuckDB. No explanation, only the SQL. Use tables and columns from the schema above.""")

        chain = prompt | llm | StrOutputParser()
        try:
            raw = chain.invoke({"schema": schema, "question": question})
        except Exception as e:
            logger.error(f"NL-to-SQL LLM error: {type(e).__name__}: {e}", exc_info=True)
            return ""

        # Extract first statement: strip and take up to semicolon or full string
        sql = raw.strip()
        if ";" in sql:
            sql = sql.split(";")[0].strip()
        # If the model wrapped in markdown code block, strip it
        if sql.startswith("```"):
            lines = sql.split("\n")
            out = []
            for line in lines:
                if line.strip() in ("```", "```sql"):
                    continue
                out.append(line)
            sql = "\n".join(out)
        return sql.strip()

    def _looks_like_sql(self, query: str) -> bool:
        """Heuristic: input is already SQL (SELECT or WITH)."""
        s = query.strip().upper()
        return s.startswith("SELECT") or s.startswith("WITH")

    def retrieve(self, query: str, k: int = 50) -> List[Document]:
        """Retrieve using a natural language question or a raw SQL SELECT.
        If query looks like SQL, it is validated (SELECT-only) and run.
        Otherwise it is converted with the LLM (schema + question) then run.
        """
        sql = query.strip()
        if not self._looks_like_sql(sql):
            sql = self._nl_to_sql(query)
            if not sql:
                logger.error("NL-to-SQL produced no SQL.")
                return []
            logger.info(f"NL-to-SQL: {sql}")

        if not self._validate_select_only(sql):
            logger.error("Rejected non-SELECT statement.")
            return []

        try:
            result = self.conn.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return []

        # Friendly display names for DuckDB auto-generated column names (e.g. COUNT(*) -> count_star())
        _COLUMN_DISPLAY_NAMES = {
            "count_star()": "Count",
            "count(*)": "Count",
        }

        documents = []
        for _, row in result.iterrows():
            parts = []
            for col, val in row.items():
                display_col = _COLUMN_DISPLAY_NAMES.get(col) or col
                parts.append(f"{display_col}: {val}")
            content = " | ".join(parts)
            doc = Document(
                page_content=content,
                metadata={"retrieval_source": "sql", "query": sql},
            )
            documents.append(doc)
        return documents[:k]

    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
