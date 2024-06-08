from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()


def get_profile_url_tavily(name:str):
    """Searches for linkedin or Twitter profile page."""
    search = TavilySearchResults()
    response = search.invoke(f'{name}')
    return response[0]['url']




