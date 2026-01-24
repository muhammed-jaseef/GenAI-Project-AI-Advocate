
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

if __name__ == "__main__":
    try:
        # Create the tool
        search_tool = get_web_search_tool(k=3)

        # Test query
        query = "The population in india"

        print("Testing Tavily Web Search Tool...")
        print(f"Query: {query}\n")

        # Run the search
        results = search_tool.invoke(query)

        # Print results
        print("Search Results:\n")

        for i, result in enumerate(results, start=1):
            print("Content:", result.get("content", "")[:50], "...")  # first 200 chars
            #print("-" * 50)

        print("\n✅ Tool is working properly!")

    except Exception as e:
        print("\n❌ Error while testing Tavily tool:")
        print(e)














