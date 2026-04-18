import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent

load_dotenv()

async def main():
    # 1. Initialize the Multi-Server Client
    # Ensure mathserver.py and weather.py are in the same directory
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio"
            },
            "weather": {
                "command": "python",
                "args": ["weather.py"],
                "transport": "stdio" 
            }
        }
    ) as mcp_client:

        # 2. Setup the Model via OpenRouter
        model = ChatOpenAI(
            api_key="Enter API key here", 
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3-8b-instruct"
        )

        # 3. Get tools and create the agent
        tools = await mcp_client.get_tools()
        agent = create_react_agent(model, tools)

        # 4. Math Request
        math_input = {"messages": [("user", "What is (3+5) x 12?")]}
        math_result = await agent.ainvoke(math_input)
        print("--- Math Query ---")
        print("Response:", math_result["messages"][-1].content)

        #print("\n" + "-"*20 + "\n")

        # 5. Weather Request
        weather_input = {"messages": [("user", "What is the weather in Bangalore?")]}
        weather_result = await agent.ainvoke(weather_input)
        print("--- Weather Query ---")
        print("Response:", weather_result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
