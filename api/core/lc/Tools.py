import ast
import astor
import copy
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, Optional
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import pandas as pd

from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.chains.llm import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.tools.base import ToolException
from langchain.tools.python.tool import PythonAstREPLTool, sanitize_input

from langchain import SQLDatabase
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

from core.callback_handler.agent_loop_gather_callback_handler import AgentLoopGatherCallbackHandler

from core.lc.nc import get_db_uri
from core.lc.prompt import single_prompt, multi_prompt, desc_prompt_template


def _get_prompt_and_tools(model_instance, conversation_message_task, rest_tokens: int):
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

        llm_callback = AgentLoopGatherCallbackHandler(
            model_instant=model_instance,
            conversation_message_task=conversation_message_task
        )
        llm = copy.deepcopy(model_instance.client)
        llm.callbacks = [llm_callback]
        prefix = desc_prompt_template.format(df_head=str(
            df.head().to_markdown()), df_describe=str(df.describe().to_markdown()))
        desc_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=prefix),
            ("human", "{question}"),
        ])
        describe_tool = Tool(name="describe_tool", func=LLMChain(llm=llm, prompt=desc_prompt).run,
                             description="useful for when you need to describe basic info of data. Input should be the human's original question.", args_schema=DescribeInput, return_direct=True)

        tools = [
            describe_tool,
            query_sql_database_tool,
            # PythonAstREPLTool(
            #     name="pandas_tool", description="import Pandas and analysis dataframe", locals={"df": df}),
            PandasTool(locals={"df": df}),
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
        ]

        return prompt, tools


class DescribeInput(BaseModel):
    question: str = Field(description="targeted question")


class PandasTool(BaseTool):
    name = "pandas_tool"
    description = (
        "Use this to import Pandas and analysis dataframe. "
        "Input should be a valid python command. "
        "1. Prepare: Preprocessing and cleaning data if necessary "
        "2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.) "
        "3. Analyze: Conducting the actual analysis (if the user asks to create a chart save it to an image and do not show the chart.)"
    )
    # return_direct = True
    handle_tool_error = True
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True

    def _run(self, query: str, **kwargs):
        try:
            if self.sanitize_input:
                query = sanitize_input(query)

            plot_func = {"plot", "line", "pie", "bar", "barh",
                         "hist", "scatter", "area", "box", "kde", "show"}
            tree = ast.parse(query)
            new_body = []
            for node in tree.body:
                new_body.append(node)
                if hasattr(node, "value"):
                    value = node.value
                    if isinstance(value, ast.Call) and value.func and (isinstance(value.func, ast.Attribute) and value.func.attr in plot_func):
                        new_body.append(
                            ast.parse(f"import matplotlib.pyplot as plt"))
                        new_body.append(ast.parse(f"import io, base64"))
                        new_body.append(ast.parse(f"buf = io.BytesIO()"))
                        new_body.append(
                            ast.parse(f"plt.savefig(buf, format='png')"))
                        new_body.append(ast.parse(f"buf.seek(0)"))
                        new_body.append(ast.parse(
                            f"'![image](data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8') + ')'"))
                        self.return_direct = True
            new_tree = ast.Module(body=new_body)
            query = astor.to_source(
                new_tree, pretty_source=lambda x: "".join(x)).strip()

            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            self.return_direct = False
            raise ToolException(repr(e))


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
