from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    # This is a placeholder implementation. Replace with actual weather fetching logic.
    return f"The weather in {city} is sunny."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")