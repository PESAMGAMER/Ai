import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()  # โหลดตัวแปรจาก .env

hf_token = os.getenv("HF_AUTH_TOKEN")

qa = pipeline(
    "question-answering",
    model="./trained_qa_model",
    tokenizer="./trained_qa_model"
)

def get_answer(question, context):
    result = qa(question=question, context=context)
    return result["answer"]