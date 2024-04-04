FROM python:3.10

WORKDIR /app

COPY ./LLM_Custom.py .
COPY ./upload_PDFAPI_chat.py .
COPY ./*.gguf .
COPY ./requirements.txt .

RUN pip install -r requirements.txt
