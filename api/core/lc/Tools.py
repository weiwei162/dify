import json
from typing import Type
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.tools.python.tool import PythonAstREPLTool

from langchain import SQLDatabase
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

from core.lc.schema import PlotConfig
from core.lc.render import draw_plotly
from core.lc.nc import get_db_uri
from core.lc.prompt import single_prompt, multi_prompt, desc_prompt_template


def _get_prompt_and_tools(llm, conversation_message_task, rest_tokens: int, callbacks: Callbacks = None):
    try:
        import pandas as pd

        pd.set_option("display.max_columns", None)
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    uri = get_db_uri(conversation_message_task.inputs['ds_id'])
    db = SQLDatabase.from_uri(
        uri,
        include_tables=conversation_message_task.inputs['include_tables'].split(
            ',') if conversation_message_task.inputs['include_tables'] else None,
        view_support=True,
    )
    _table_names = db.get_usable_table_names()
    table_names = ", ".join(_table_names)

    table_info = db.get_table_info_no_throw(_table_names)

    query_sql_database_tool_description = (
        "Input to this tool is a detailed and correct SQL query, output is a "
        "result from the database. If the query is not correct, an error message "
        "will be returned. If an error is returned, rewrite the query, check the "
        "query, and try again. If you encounter an issue with Unknown column "
        f"'xxxx' in 'field list', checking tables "
        "to query the correct table fields."
    )
    query_sql_database_tool = QuerySQLDataBaseTool(
        db=db, description=query_sql_database_tool_description
    )

    if len(_table_names) == 1:
        engine = create_engine(uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text(f"SELECT * FROM {_table_names[0]}"), conn)

        prefix = single_prompt.format(
            dialect=db.dialect, top_k=10, table_info=table_info, df_head=str(df.head().to_markdown()))
        prompt = SystemMessage(content=prefix)

        if not conversation_message_task.streaming:
            tools = [SQLTool(uri=uri)]
            return prompt, tools

        desc_prompt = PromptTemplate.from_template(desc_prompt_template).partial(
            df_head=str(df.head().to_markdown()), df_describe=str(df.describe().to_markdown()))
        describe_tool = Tool(name="describe_tool", func=LLMChain(llm=llm, prompt=desc_prompt).run,
                             description="useful for when you need to describe basic info of data", args_schema=DescribeInput, return_direct=True)

        tools = [
            describe_tool,
            query_sql_database_tool,
            PythonAstREPLTool(
                name="pandas_tool", description="import Pandas and analysis dataframe", locals={"df": df}),
            PlotTool(uri=uri),
        ]

        return prompt, tools

    elif len(_table_names) > 1:
        prefix = multi_prompt.format(
            dialect=db.dialect, top_k=10, table_names=table_names)
        prompt = SystemMessage(content=prefix)

        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "argument key is 'table_names' "
            "Example Input: 'table1, table2, table3'"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=db, description=info_sql_database_tool_description
        )

        if not conversation_message_task.streaming:
            tools = [info_sql_database_tool, SQLTool(uri=uri)]
            return prompt, tools

        tools = [
            info_sql_database_tool,
            query_sql_database_tool,
            PlotTool(uri=uri),
        ]

        return prompt, tools


class DescribeInput(BaseModel):
    question: str = Field(description="targeted question")


class PlotTool(BaseTool):
    name = "generate_chart"
    description = """
    Generate the chart with given parameters"""
    args_schema: Type[BaseModel] = PlotConfig

    return_direct = True

    uri: str

    def _run(self, **kwargs):
        sql = kwargs["sql"]
        engine = create_engine(self.uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)

        config = PlotConfig.parse_obj(kwargs)
        figure = draw_plotly(df, config, False)
        return "![image]({})".format(figure)


class SQLTool(BaseTool):
    """Tool for querying a SQL database."""

    name = "sql_chain"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    return_direct = True

    uri: str

    def _run(self, query: str, **kwargs):
        """Execute the query, return the results or an error message."""
        engine = create_engine(self.uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        return json.dumps({"sql": query, "data": df.to_dict('records')})
