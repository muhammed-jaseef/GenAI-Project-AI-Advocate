
from langchain_community.tools.tavily_search import TavilySearchResults
from configure import TAVILY_API_KEY


def get_web_search_tool(k: int = 3):
    """
    Create Tavily web search tool.

    Args:
        k: Number of search results to return

    Returns:
        TavilySearchResults tool
    """
    return TavilySearchResults(
        api_key=TAVILY_API_KEY,
        k=k,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )

















