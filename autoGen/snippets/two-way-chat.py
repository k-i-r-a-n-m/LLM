import os
from pprint import pprint
from dotenv import load_dotenv
from autogen import ConversableAgent

load_dotenv()

llm_config={
            "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0, "api_key": os.environ.get("OPENAI_API_KEY")}]}

def singleAgent():
    agent = ConversableAgent(
        "chatbot",
        llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}]},
        code_execution_config=False,  # Turn off code execution, by default it is off.
        function_map=None,  # No registered functions, by default it is None.
        human_input_mode="NEVER",  # Never ask for human input.
    )

    replay = agent.generate_reply(messages=[{"role": "user", "content": "Tell me a joke!"}])
    print(replay)


def twoAgent():
    cathy = ConversableAgent(
        "cathy",
        system_message="Your name is Cathy and you are a part of a duo of comedians.",
        llm_config=llm_config,
        human_input_mode="NEVER",  # Never ask for human input.
        # max_consecutive_auto_reply=2
    )

    joe = ConversableAgent(
        "joe",
        system_message="Your name is Joe and you are a part of a duo of comedians.",
        llm_config=llm_config,
        human_input_mode="NEVER",  # Never ask for human input.
        max_consecutive_auto_reply=2
    )

    # chat_result = joe.initiate_chat(cathy,message="hi cathy tell me a joke!")
    chat_result = joe.initiate_chat(cathy,message="hi cathy tell me a joke!")
    pprint(chat_result)


def twoAgent_teachingAss():
    teacher = ConversableAgent(
        "teacher",
        system_message="""
        you ask question to student .
        you are professional math teacher.
        you help children learn maths with easy step by step understanding.
        you don't give the answer right away help student to derive it. 
        if student couldn't come up with the answer give the answer and end with TERMINATE message in new line.
         """,
        llm_config=llm_config,
        human_input_mode="NEVER",  # Never ask for human input.
        # max_consecutive_auto_reply=2
        is_termination_msg=lambda msg: "terminate" in msg["content"].lower()
    )

    student = ConversableAgent(
        "student",
        # system_message="""
        # you ask question to teacher to solve problem and you don't have to judge whether it is correct or not you should only learn from the teacher
        #  don't jump to answer on you own.
        # you can get help from teacher if necessary or couldn't get the answer.
        # """,
        system_message="""
        you are a student who is learning maths by practicing.
        You should only try to solve the given problem or question.
         you know nothing about maths.
         you ask teacher for only assistance and not for answer.
        if you can't solve problems/questions given. give up and ask for answer as last resort.
        don't be appreciative.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",  # Never ask for human input.
        # max_consecutive_auto_reply=2
        is_termination_msg=lambda msg: "terminate" in msg["content"].lower()
    )
    result = student.initiate_chat(teacher,message="give me some moderate math problem to solve")
    # result = teacher.initiate_chat(student,message="difference between rhomboid and cube?",max_turns=5)
    pprint(result)


def coder_userProxyAgent():
    coder = ConversableAgent(
        name="programmer",
        system_message="""
        you are a intelligent programmer who like to code.
        you write code for the given problem.
        you build solutions that are efficient.
        you should only give the code without comments
        """,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    result = coder.generate_reply(messages=[{"role": 'user',"content": 'write a program to in python to get data from the druid'}])
    pprint(result)

def always_human_input_agent():
    agent_with_number = ConversableAgent(
        "agent_with_number",
        system_message="You are playing a game of guess-my-number. You have the "
                       "number 53 in your mind, and I will try to guess it. "
                       "If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
        llm_config=llm_config,
        is_termination_msg=lambda msg: "53" in msg["content"],  # terminate if the number is guessed by the other agent
        human_input_mode="NEVER",  # never ask for human input
    )

    human_proxy = ConversableAgent(
        "human_proxy",
        llm_config=False,  # no LLM used for human proxy
        human_input_mode="ALWAYS",  # always ask for human input
        # is_termination_msg=lambda msg: "53" in msg["content"]
    )

    # Start a chat with the agent with number with an initial guess.
    result = human_proxy.initiate_chat(
        agent_with_number,  # this is the same agent with the number as before
        message="10",
    )

    pprint(result)


def terminate_human_input_agent():
    agent_with_number = ConversableAgent(
        "agent_with_number",
        system_message="You are playing a game of guess-my-number. "
                       "In the first game, you have the "
                       "number 53 in your mind, and I will try to guess it. "
                       "If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
        llm_config=llm_config,
        max_consecutive_auto_reply=1,  # maximum number of consecutive auto-replies before asking for human input
        is_termination_msg=lambda msg: "53" in msg["content"],  # terminate if the number is guessed by the other agent
        human_input_mode="TERMINATE",  # ask for human input until the game is terminated
    )

    agent_guess_number = ConversableAgent(
        "agent_guess_number",
        system_message="I have a number in my mind, and you will try to guess it. "
                       "If I say 'too high', you should guess a lower number. If I say 'too low', "
                       "you should guess a higher number. ",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    result = agent_with_number.initiate_chat(
        agent_guess_number,
        message="I have a number between 1 and 100. Guess it!",
    )

    pprint(result)


if __name__ == "__main__":
    # singleAgent()
    # twoAgent()
    # twoAgent_teachingAss()
    # coder_userProxyAgent()
    # always_human_input_agent()
    terminate_human_input_agent()