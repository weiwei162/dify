from typing import List, Tuple, Any, Union, Sequence, Optional

from langchain import BasePromptTemplate
from langchain.agents import StructuredChatAgent, AgentOutputParser, Agent
from langchain.agents.structured_chat.base import HUMAN_MESSAGE_TEMPLATE
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.memory.summary import SummarizerMixin
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain.agents.structured_chat.prompt import PREFIX, SUFFIX

from core.agent.agent.calc_token_mixin import CalcTokenMixin, ExceededLLMTokensLimitError


FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
The nouns in the format of "Thought", "Action", "Action Input", "Final Answer" must be expressed in English.
Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""


class AutoSummarizingStructuredChatAgent(StructuredChatAgent, CalcTokenMixin):
    moving_summary_buffer: str = ""
    moving_summary_index: int = 0
    summary_llm: BaseLanguageModel

    def should_use_agent(self, query: str):
        """
        return should use agent
        Using the ReACT mode to determine whether an agent is needed is costly,
        so it's better to just use an Agent for reasoning, which is cheaper.

        :param query:
        :return:
        """
        return True

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)

        prompts, _ = self.llm_chain.prep_prompts(input_list=[self.llm_chain.prep_inputs(full_inputs)])
        messages = []
        if prompts:
            messages = prompts[0].to_messages()

        rest_tokens = self.get_message_rest_tokens(self.llm_chain.llm, messages)
        if rest_tokens < 0:
            full_inputs = self.summarize_messages(intermediate_steps, **kwargs)

        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    def summarize_messages(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs):
        if len(intermediate_steps) >= 2:
            should_summary_intermediate_steps = intermediate_steps[self.moving_summary_index:-1]
            should_summary_messages = [AIMessage(content=observation)
                                       for _, observation in should_summary_intermediate_steps]
            if self.moving_summary_index == 0:
                should_summary_messages.insert(0, HumanMessage(content=kwargs.get("input")))

            self.moving_summary_index = len(intermediate_steps)
        else:
            error_msg = "Exceeded LLM tokens limit, stopped."
            raise ExceededLLMTokensLimitError(error_msg)

        summary_handler = SummarizerMixin(llm=self.summary_llm)
        if self.moving_summary_buffer and 'chat_history' in kwargs:
            kwargs["chat_history"].pop()

        self.moving_summary_buffer = summary_handler.predict_new_summary(
            messages=should_summary_messages,
            existing_summary=self.moving_summary_buffer
        )

        if 'chat_history' in kwargs:
            kwargs["chat_history"].append(AIMessage(content=self.moving_summary_buffer))

        return self.get_full_inputs([intermediate_steps[-1]], **kwargs)

    @classmethod
    def from_llm_and_tools(
            cls,
            llm: BaseLanguageModel,
            tools: Sequence[BaseTool],
            callback_manager: Optional[BaseCallbackManager] = None,
            output_parser: Optional[AgentOutputParser] = None,
            prefix: str = PREFIX,
            suffix: str = SUFFIX,
            human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
            format_instructions: str = FORMAT_INSTRUCTIONS,
            input_variables: Optional[List[str]] = None,
            memory_prompts: Optional[List[BasePromptTemplate]] = None,
            **kwargs: Any,
    ) -> Agent:
        return super().from_llm_and_tools(
            llm=llm,
            tools=tools,
            callback_manager=callback_manager,
            output_parser=output_parser,
            prefix=prefix,
            suffix=suffix,
            human_message_template=human_message_template,
            format_instructions=format_instructions,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
            **kwargs,
        )