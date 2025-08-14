import os
from tools.dspy_wiki_rag import dspy_wiki_rag, dspy_wiki_search
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory

def init_lm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    """Initialize the language model"""
    llm = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature)
    return llm

def init_tools():
    """Initialize the tools"""
    tools = [Tool(name="dspy_wiki_search", func=dspy_wiki_search, description="Search Wikipedia for information"), Tool(name="dspy_wiki_rag", func=dspy_wiki_rag, description="Answer a question using the WikiRAG pipeline")]
    return tools

def init_agent(llm: ChatOpenAI):
    """Initialize the agent"""
    memory = ConversationBufferWindowMemory(k=1, return_messages=True)
    tools = init_tools()
    agent = initialize_agent(
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # avoids stop param for some models
        llm=llm,
        verbose=True,
        memory=memory,
        max_iterations=4,
        early_stopping_method="generate",
    )
    return agent

if __name__ == "__main__":
    llm = init_lm()
    agent = init_agent(llm)
    input_text = input("Enter a question: ")
    agent.run(input_text)