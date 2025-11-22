from custom_utils.tools import TOOLS
import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from langchain.agents import create_agent

system_prompt = """You are a helpful assistant with access to the following tools:

- match_score_tool: Get an overall match score between the cand and a job.
- read_file_tool: Read the content of a file.

Use these tools to analyse resumes and provide insights.
"""
system_message = """
You must never pass tool names as arguments to other tools.
When a tool requires the content of a resume or job description,

"""



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=system_message
)


