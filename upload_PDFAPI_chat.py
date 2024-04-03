from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from secrets import token_hex
import uvicorn

import PyPDF2
from io import BytesIO
import requests
import os

import LLM_Custom

class Input(BaseModel):
    input: str | None = None

app = FastAPI(title="PDF Upload")

MODEL_PATH = "/mnt/c/Users/Danny/Downloads/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    
model = LLM_Custom.create_model(MODEL_PATH)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global filename
    contents = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    if os.path.exists(file.filename):
        filename = file.filename
        return {"status_code": 200, "message": "good"}
    else:
        raise HTTPException(status_code=500, detail="not good")
    
@app.post("/chat")
async def process_pdf(input: Input):
    global filename
    # Process the input as needed
    # Return the name of the last uploaded PDF

    agent_chain = await invoke_chat(filename)
    reply = agent_chain.invoke({"input": {input.input}})
    return reply

async def invoke_chat(pdf_source):
    tools = LLM_Custom.create_tools(pdf=pdf_source, path=MODEL_PATH, model=model)
    agent_chain = LLM_Custom.create_agent(model=model, tools=tools)
    
    return agent_chain
   

if __name__ == "__main__":
    uvicorn.run("upload_PDFAPI_chat:app", host="127.0.0.1", reload=True)