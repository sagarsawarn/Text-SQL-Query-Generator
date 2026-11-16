from langchain_core.prompts import ChatPromptTemplate

template = """
You are a SQL query generator\n
Given a database schema and a natural language question, generate only the corresponding SQL statement—nothing else.\n
If the question cannot be converted into a valid SQL query or is unrelated to the provided schema, respond with exactly: "I don't know".\n\n
Rules:\n
- Use ONLY the table and column names listed in Table Info. Never invent columns such as ids or joins that are not listed.\n
- When filtering on text, use a value that appears in the listed distinct values or sample rows. If the user's wording differs (e.g. "Bangalore" vs "Bengaluru"), pick the closest listed value, or use LIKE with a short keyword.\n
- Compare text case-insensitively (e.g. LOWER(column) = LOWER('value') or LIKE).\n
- Prefer a single simple SELECT on the table; avoid subqueries unless the question needs them.\n\n


Table Info: {table_info}\n\n
User question: {query}\n\n
Output (SQL only):
"""
prompt = ChatPromptTemplate.from_template(template)

explanation_prompt = ChatPromptTemplate.from_template(
	"""
Explain the following SQL query in one short, beginner-friendly sentence.
Do not rewrite the query or include markdown.

SQL query:
{query}
"""
)

# Feature: self-correcting SQL. Used when a generated query fails to run —
# the error is sent back to the model so it can produce a corrected version.
fix_prompt = ChatPromptTemplate.from_template(
	"""
The following SQL query failed when run against the database.

Table Info: {table_info}

Original question: {query}

SQL that failed:
{bad_sql}

Error message:
{error}

Fix the SQL so it runs correctly against the schema above. Output ONLY the corrected SQL query, nothing else.
"""
)
