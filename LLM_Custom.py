import os
import io
import requests
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain

from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, create_react_agent
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv


#MODEL_PATH = "/mnt/c/Users/Danny/Downloads/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

def create_model(path):
    model = LlamaCpp(
            model_path=path,
            temperature=0.75,
            n_gpu_layers = -1,
            n_batch = 512,
            max_tokens=2000,
            top_p=1,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=4096,
        )
    return model

def process_pdf(pdf: str, path: str):
    loader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(loader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    chunks = text_splitter.split_text(raw_text)

    knowledgeBase = Chroma.from_texts(chunks, LlamaCppEmbeddings(model_path=path))

    return knowledgeBase

def top5_results(query):
    load_dotenv()
    search = GoogleSearchAPIWrapper()
    return search.results(query, 5)

def create_tools(pdf: str, path: str, model: any):
    result = process_pdf(pdf, path)
    search_result = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=result.as_retriever()
    )
    tools = [Tool(
        name="PDF Search",
        func=search_result.run,
        description="Generate texts from PDF.",
        )]
    tools += [Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=top5_results,
        )]
    return tools


def create_prompt_react():
    template = '''Answer the following questions as best you can. You have access to the following chat history and tools:

    {chat_history}
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be based of the [{chat_history}] and one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat at maximum 5 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    return prompt

def create_agent(model: any, tools: any):
    memory = ConversationBufferMemory(memory_key="chat_history")
    #prompt = create_prompt()
    prompt = create_prompt_react()
    llm_chain = LLMChain(llm=model, prompt=prompt)
    #agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
        max_iterations=3
    )

    return agent_chain