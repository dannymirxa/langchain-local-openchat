from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi_htmx import htmx, htmx_init

from secrets import token_hex
import uvicorn

import PyPDF2
from io import BytesIO
import requests

app = FastAPI(title="PDF Upload")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)
     
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    
    # Process saved file
    return await process_pdf(file.filename, is_local_file=True)   

async def process_pdf(pdf_source, is_local_file=False):
    # Process the PDF from URL or local file
    file = BytesIO(requests.get(pdf_source).content) if not is_local_file else open(pdf_source, 'rb')

    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_number in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_number].extract_text()

    if is_local_file:
        file.close()
    return text

if __name__ == "__main__":
    uvicorn.run("upload_PDFAPI:app", host="127.0.0.1", reload=True)