import json
from typing import Optional, Tuple
from pandas import DataFrame

from langchain.agents import AgentExecutor
from langchain.schema import (
    AgentAction,
    AgentFinish,
)


class LCAgentExecutor(AgentExecutor):
    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )

        if isinstance(observation, DataFrame):
            content = str(observation.to_markdown())
        elif not isinstance(observation, str):
            try:
                content = json.dumps(observation, ensure_ascii=False)
            except Exception:
                content = str(observation)
        else:
            content = observation
        if len(content) > 2000:
            return AgentFinish({self.agent.return_values[0]: content}, "",)

        return None
