import os
from dotenv import load_dotenv
from autogen.agentchat.contrib.agent_builder import AgentBuilder
import autogen


load_dotenv()

llm_config={
            "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0.6, "api_key": os.environ.get("OPENAI_API_KEY")}]}


config_file_or_env = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
)

builder = AgentBuilder(config_file_or_env="OAI_CONFIG_LIST.json", builder_model='gpt-3.5-turbo', agent_model='gpt-3.5-turbo')


# building_task = """
#      plan what needs to be done and verify the plan thoroughly and deduce a single plane which is suitable.
#      using the plan produce a step by step instruction and write a python program and execute it .
#      save the result in a json file.
#     """

building_task = """ 
    solve only using .env variables as input.
    """

# building_task = "Find a paper on arxiv by programming, and analyze its application in some domain. For example, find a latest paper about gpt-4 on arxiv and find its potential applications in software."

agent_list, agent_configs = builder.build(building_task, llm_config, coding=True)



def start_task(execution_task: str, agent_list: list, llm_config: dict):
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

start_task(
    # execution_task=""""
    #  reported an anomaly from prometheus metrics.
    #  verify whether the anomaly is present or not from the druid.
    #  """,
    execution_task=""""
      subtract b-a
     """,
    agent_list=agent_list,
    llm_config=llm_config
)