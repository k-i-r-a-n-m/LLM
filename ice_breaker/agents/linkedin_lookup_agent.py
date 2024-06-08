
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor , create_react_agent
from langchain import hub
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from tools.tools import get_profile_url_tavily

load_dotenv()


def linkedin_lookup_agent(name:str):
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

    template = """given the full name {name_of_person} I want you to get me a link to their linkedin profile page. also instagram profile link
                your answer should contain only a URLs of linkedin and instagram"""
    
    prompt_template = PromptTemplate(template=template, input_variables=['name_of_person'])

    tools_for_agent = [
        Tool(
            name='Crawl Google to get Linkedin profile page',
            func=get_profile_url_tavily,
            description="useful when you need to get the linkedin page URL and instagram profile link"
        )
    ]

    react_prompt = hub.pull('hwchase17/react')  # user/templateName

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(input={
        "input": prompt_template.format_prompt(name_of_person=name) # guess: "input" must be template from the hub
        
    })

    linkedIn_profile_url = result['output']

    return linkedIn_profile_url


print(linkedin_lookup_agent("santhosh c edify salem"))

