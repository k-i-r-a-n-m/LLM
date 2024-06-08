from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from typing import Union, List, Tuple
from langchain_core.agents import AgentAction, AgentFinish


@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def format_log_to_str(intermediate_steps: List[Tuple[AgentAction, str]], observation_prefix: str = "Observation:",
                      llm_prefix: str = "Thought:") -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


def find_tool_by_name(tools, tool_name):
    for tool in tools:
        if tool.name == tool_name:
            return tool


if __name__ == '__main__':
    load_dotenv()

    template = """ 
           Answer the following questions as best you can. You have access to the following tools:
           {tools}

           Use the following format:

           Question: the input question you must answer
           Thought: you should always think about what to do
           Action: the action to take, should be one of [{tool_names}]
           Action Input: the input to the action
           Observation: the result of the action
           ... (this Thought/Action/Action Input/Observation can repeat N times)
           Thought: I now know the final answer
           Final Answer: the final answer to the original input question

           Begin!

           Question: {input}
           Thought:{agent_scratchpad} 
           """

    intermediate_steps = []
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [get_text_length]

    prompt = PromptTemplate \
        .from_template(template=template) \
        .partial(tools=render_text_description(tools),
                 tool_names=', '.join([t.name for t in tools]))  # LLm only accepts STRING as input

    agent = {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
            } | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "what is the length of dog in characters ?", "agent_scratchpad": intermediate_steps})

    if (isinstance(agent_step, AgentAction)):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name).func
        tool_input = agent_step.tool_input

        observation = tool_to_use(tool_input)
        print(f"{observation=}")
        intermediate_steps.append((agent_step, observation))

    # print(intermediate_steps)

    print(agent_step)
