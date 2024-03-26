import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentExecutor, Tool, ZeroShotAgent, create_react_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from htmlTemplates import css, bot_template, user_template

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
MODEL_PATH = "/mnt/c/Users/Danny/Downloads/openchat-3.5-0106.Q5_K_S.gguf"

def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    return text_splitter.split_text(raw_text)

def get_vector_store(text_chunks) -> FAISS: 
    llama_embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, 
                                          n_ctx = 4096,
                                          n_gpu_layers=n_gpu_layers,
                                          n_batch=n_batch)
    return FAISS.from_texts(texts = text_chunks, embedding = llama_embeddings)

def get_conversation_chain(vector_store : FAISS):
    llm = LlamaCpp (
        model_path=MODEL_PATH,
        temperature=0,
        n_ctx=4096,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=2000,
        top_p=1,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
    )
    return conversation_chain

def handle_userinput(user_questions : str):
    response = st.session_state.conversation({"question": user_questions})
    st.session_state.chat_history =  response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    
    st.set_page_config(page_title="Openchat 3.5", page_icon="🧠")

    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Openchat 3.5 🧠")
    user_questions = st.text_input("Just ask/ say anything please:")
    if user_questions:
        handle_userinput(user_questions)

    st.write(user_template.replace("{{MSG}}", "Hello gundam"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello man"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdf and press Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #get text chnks
                text_chunks = get_text_chunks(raw_text)

                #get vector store
                vector_store = get_vector_store(text_chunks)
                
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()