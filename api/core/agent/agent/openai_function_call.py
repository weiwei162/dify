import json
from json import JSONDecodeError
from typing import List, Tuple, Any, Union, Sequence, Optional

from langchain.agents import OpenAIFunctionsAgent, BaseSingleActionAgent
from langchain.agents.openai_functions_agent.base import _format_intermediate_steps, _FunctionsAgentAction
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import AgentAction, AgentFinish, SystemMessage, AIMessage, BaseMessage, OutputParserException
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

from core.agent.agent.calc_token_mixin import ExceededLLMTokensLimitError
from core.agent.agent.openai_function_call_summarize_mixin import OpenAIFunctionCallSummarizeMixin


class AutoSummarizingOpenAIFunctionCallAgent(OpenAIFunctionsAgent, OpenAIFunctionCallSummarizeMixin):

    @classmethod
    def from_llm_and_tools(
            cls,
            llm: BaseLanguageModel,
            tools: Sequence[BaseTool],
            callback_manager: Optional[BaseCallbackManager] = None,
            extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
            system_message: Optional[SystemMessage] = SystemMessage(
                content="You are a helpful AI assistant."
            ),
            **kwargs: Any,
    ) -> BaseSingleActionAgent:
        return super().from_llm_and_tools(
            llm=llm,
            tools=tools,
            callback_manager=callback_manager,
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message,
            **kwargs,
        )

    def should_use_agent(self, query: str):
        """
        return should use agent

        :param query:
        :return:
        """
        original_max_tokens = self.llm.max_tokens
        self.llm.max_tokens = 15

        prompt = self.prompt.format_prompt(input=query, agent_scratchpad=[])
        messages = prompt.to_messages()

        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=None
        )

        function_call = predicted_message.additional_kwargs.get("function_call", {})

        self.llm.max_tokens = original_max_tokens

        return True if function_call else False

    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()

        # summarize messages if rest_tokens < 0
        try:
            messages = self.summarize_messages_if_needed(messages, functions=self.functions)
        except ExceededLLMTokensLimitError as e:
            return AgentFinish(return_values={"output": str(e)}, log=str(e))

        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision

    @classmethod
    def get_system_message(cls):
        return SystemMessage(content="You are a helpful AI assistant.\n"
                                     "The current date or current time you know is wrong.\n"
                                     "Respond directly if appropriate.")

    def return_stopped_response(
            self,
            early_stopping_method: str,
            intermediate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any,
    ) -> AgentFinish:
        try:
            return super().return_stopped_response(early_stopping_method, intermediate_steps, **kwargs)
        except ValueError:
            return AgentFinish({"output": "I'm sorry, I don't know how to respond to that."}, "")


def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            if function_name == "python":
                _tool_input = {'__arg1': function_call["arguments"]}
            else:
                raise OutputParserException(
                    f"Could not parse tool input: {function_call} because "
                    f"the `arguments` is not valid JSON."
                )

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)
