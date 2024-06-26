import os
from typing import Annotated, Literal
from autogen import ConversableAgent
from dotenv import load_dotenv

from autogen import register_function
from pprint import pprint as pp
from typing import Any



load_dotenv()


llm_config={
            "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0, "api_key": os.environ.get("OPENAI_API_KEY")}]}


Operator = Literal["+", "-", "*", "/","**"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    elif operator == "**":
        return pow(a, b)
    else:
        raise ValueError("Invalid operator")


# Let's first define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message='''You are a helpful AI assistant. 
                   You can help with simple calculations. 
                   you solve problems step by step by breaking down.
                   once broken down you can use the given tools to find the right operation to perform.
                   Return 'TERMINATE' when the task is done.''',
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# METHOD:1
# ======================================
# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="calculator")(calculator)

def add(a:Any,b:Any)->Any:
    return a+b

# METHOD:2
# ======================================
# Register the calculator function to the two agents.
register_function(
    calculator,
    caller=assistant,  # The assistant agent can suggest calls to the calculator.
    executor=user_proxy,  # The user proxy agent can execute the calculator calls.
    name="calculator",  # By default, the function name is used as the tool name.
    description="A simple calculator",  # A description of the tool.
)


# Query the Agent to execute the tool
# chat_result = user_proxy.initiate_chat(assistant, message="What is (44232 + 13312 / ( 232 - 32 )) * 5?")
chat_result = user_proxy.initiate_chat(assistant,
                                       message="what is (1423 - 123) / 3 + (32 + 23) * 5?",
                                       summary_method="reflection_with_llm",) # 708
print(chat_result.summary)
# More complex than the tool
# chat_result = user_proxy.initiate_chat(assistant, message="what is 2 + 4 / (22 / 6) * 2 ?")

# chat_result = user_proxy.initiate_chat(assistant, message="how are you")


# Assistant equipped with tools and the tools are of OPEN-AI's tool API format
# pp(assistant.llm_config["tools"])


# Without the tools the ai is able to breakdown the step and calculate it
# result = assistant.generate_reply(messages=[{"role": "user", "content": "What is (44232 + 13312 / ( 232 - 32 )) * 5?"}])
# result = assistant.generate_reply(messages=[{"role": "user", "content": "what is (1423 - 123) / 3 + (32 + 23) * 5?"}])
# print(result)
