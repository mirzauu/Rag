
from langchain_openai import ChatOpenAI
import os
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Crew
from crewai import Task
from crewai import Agent

import os

# Set the API key
os.environ['GROQ_API_KEY'] = 'Add Your Groq API Key'

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

import requests

pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
response = requests.get(pdf_url)

with open('attenstion_is_all_you_need.pdf', 'wb') as file:
    file.write(response.content)

rag_tool = PDFSearchTool(
    pdf='attenstion_is_all_you_need.pdf',
    config={
        "llm": llm
    }
)

@tool("Paper QA Tool")
def paper_qa_tool(query: str) -> str:
    """Useful to answer questions about the paper Attention is All You Need"""
    return rag_tool.search(query)

researcher = Agent(
    role='Researcher',
    goal='Do deep research on a given topic using the given paper',
    backstory="""You are a Researcher, an expert in understanding complex research papers and extracting valuable information from them.""",
    verbose=True,
    allow_delegation=False,
    tools=[paper_qa_tool]
)

writer = Agent(
    role='Technical Writer',
    goal='Write a detailed article on the research paper topic',
    backstory="""You are a Technical Writer, skilled in transforming research content into clear and engaging articles for a broader audience.""",
    verbose=True,
    allow_delegation=True
)

task1 = Task(
    description="""Research and understand the key ideas, methodologies, and conclusions of the paper 'Attention is All You Need'.""",
    expected_output="Detailed notes on the key concepts, architecture, and findings of the paper.",
    tools=[paper_qa_tool],
    agent=researcher
)

task2 = Task(
    description="Write a blog post summarizing the research paper 'Attention is All You Need'.",
    expected_output="A well-structured blog post explaining the paper's ideas and importance in layman's terms.",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

result = crew.kickoff()
print(result)
