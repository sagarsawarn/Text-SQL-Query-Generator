import re
import pandas as pd
from process.prompt_process import prompt, explanation_prompt, fix_prompt
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI


REQUEST_TIMEOUT_SECONDS = 45

# The AI explanation is a second, separate model call after the SQL is
# generated, which roughly doubles the wait. Set to True to bring it back;
# otherwise the instant local explainer (explain_sql) is used.
USE_AI_EXPLANATION = False


def clean_sql(sql_code):
    clean_sql_code = re.sub(r"```(?:sqlite|sql)?\n(.*?)\n```", r"\1", sql_code, flags=re.DOTALL).strip()
    return clean_sql_code


def is_read_only_select(query):
    query_without_comments = re.sub(r"/\*.*?\*/|--[^\n]*", "", query, flags=re.DOTALL).strip()
    query_without_final_semicolon = query_without_comments.rstrip(";").strip()

    return (
        bool(re.match(r"(?i)^select\b", query_without_final_semicolon))
        and ";" not in query_without_final_semicolon
    )


def explain_sql(query):
    """Regex-based explanation. Kept as a fallback for get_explanation() in
    case the AI explanation call fails (quota, network, etc.)."""
    sql = clean_sql(query).rstrip(";").strip()
    match = re.match(
        r"(?is)^select\s+(.*?)\s+from\s+([\w.]+)(?:\s+where\s+(.+))?$",
        sql,
    )

    if not match:
        return "This query retrieves data from the database."

    selected_columns, table_name, where_clause = match.groups()
    column_description = "all columns" if selected_columns == "*" else selected_columns
    explanation = f"This query selects {column_description} from the {table_name} table"

    if where_clause:
        conditions = re.split(r"(?i)\s+(AND|OR)\s+", where_clause)
        formatted_conditions = []
        for condition in conditions:
            if condition.upper() in {"AND", "OR"}:
                formatted_conditions.append(condition.lower())
                continue

            condition_match = re.match(
                r"\s*([\w.]+)\s*(>=|<=|<>|!=|=|>|<)\s*('.*?'|\".*?\"|[^\s]+)\s*",
                condition,
            )
            if not condition_match:
                formatted_conditions.append(condition.strip())
                continue

            column, operator, value = condition_match.groups()
            value = value.strip("'\"")
            operator_text = {
                "=": "is",
                "!=": "is not",
                "<>": "is not",
                ">": "is greater than",
                "<": "is less than",
                ">=": "is greater than or equal to",
                "<=": "is less than or equal to",
            }[operator]
            formatted_conditions.append(f"the {column} {operator_text} {value}")

        explanation += " where " + " ".join(formatted_conditions)

    return explanation + "."


MAX_DISTINCT_VALUES = 25
MAX_VALUE_LENGTH = 40
SAMPLE_ROWS = 5


def build_table_info(engine):
    """Describe every table with its columns, sample rows and, for columns
    with few distinct values, the real values stored in them. Without the real
    values the model has to guess (e.g. 'Engineering' vs 'Software'), which
    makes queries run fine but return zero rows."""
    db = SQLDatabase(engine, sample_rows_in_table_info=SAMPLE_ROWS)
    parts = [db.get_table_info()]

    with engine.connect() as conn:
        for table in db.get_usable_table_names():
            df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            lines = [f'Row count of "{table}": {len(df)}']
            for column in df.columns:
                series = df[column].dropna()
                if series.empty:
                    continue
                if series.dtype == object:
                    distinct = series.astype(str).unique()
                    if len(distinct) <= MAX_DISTINCT_VALUES:
                        values = ", ".join(repr(v[:MAX_VALUE_LENGTH]) for v in distinct)
                        lines.append(f'Distinct values of "{column}": {values}')
                elif pd.api.types.is_numeric_dtype(series):
                    lines.append(f'Range of "{column}": {series.min()} to {series.max()}')
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


class Text2SQL():
    def __init__(self, model_info, question="Get all row?"):
        self.model = {
            # A request timeout stops the UI spinner from hanging forever when
            # the provider is slow or the connection stalls.
            "Gemini": lambda: ChatGoogleGenerativeAI(model=model_info["model_name"], google_api_key=model_info["api_key"], max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS, transport="rest", thinking_budget=0),
            "GPT": lambda: ChatOpenAI(model=model_info["model_name"], api_key=model_info["api_key"], timeout=REQUEST_TIMEOUT_SECONDS, max_retries=1),
        }[model_info["model_type"]]
        self.question = question
        # Filled in by get_answer()/init_db(); used later by fix_query() so a
        # failed SQL query can be corrected without the caller having to keep
        # track of the schema text separately.
        self.table_info = ""

    def init_db(self, data_files):
        self.engine = create_engine("sqlite:///:memory:")

        for file_index, data_file in enumerate(data_files):
            table_name, extension = data_file.name.rsplit(".", 1)

            if extension == "csv":
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)

            table_name = "data" if file_index == 0 else table_name
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)

        table_info = build_table_info(self.engine)
        return self.get_answer(f"The primary uploaded table is named data.\n\n{table_info}")

    def get_answer(self, data_info):
        self.table_info = data_info
        chain = prompt | self.model()
        response = chain.invoke({"table_info": data_info, "query": self.question})

        return clean_sql(response.content)

    def get_explanation(self, query):
        """Feature: real AI-generated explanation instead of a regex guess.
        Falls back to the local regex explainer if the model call fails."""
        if not USE_AI_EXPLANATION:
            return explain_sql(query)
        try:
            chain = explanation_prompt | self.model()
            response = chain.invoke({"query": query})
            return response.content.strip()
        except Exception:
            return explain_sql(query)

    def fix_query(self, data_info, bad_sql, error):
        """Feature: self-correcting SQL. Sends the failed query and its error
        back to the model and returns a corrected query."""
        chain = fix_prompt | self.model()
        response = chain.invoke(
            {
                "table_info": data_info or self.table_info,
                "query": self.question,
                "bad_sql": bad_sql,
                "error": str(error),
            }
        )
        return clean_sql(response.content)

    def run_query(self, query):
        if not is_read_only_select(query):
            raise ValueError("Only a single read-only SELECT query can be previewed.")

        with self.engine.connect() as conn:
            get_data = pd.read_sql(query, conn)

        return get_data
