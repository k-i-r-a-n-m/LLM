from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

tool = TavilySearchResults()

# result = tool.invoke({"query": "What happened in the latest burning man floods"})
result = tool.run(f'recent volcano burst')

print(result)