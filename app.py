from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import re

# =========================
# APP INIT
# =========================
app = FastAPI(title="Text Summarizer App")

# =========================
# CORS (FOR FRONTEND)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_summary_model")

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model folder not found at: {MODEL_PATH}"
    )

# =========================
# LOAD MODEL
# =========================
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

# =========================
# DEVICE SETUP
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model.to(device)

# =========================
# REQUEST BODY
# =========================
class DialogueInput(BaseModel):
    dialogue: str

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# SUMMARIZATION FUNCTION
# =========================
def summarize_text(text: str):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return HTMLResponse("""
    <h1>AI Text Summarizer API is running 🚀</h1>
    <p>Go to <a href="/docs">/docs</a> to test API</p>
    """)

# =========================
# API ENDPOINT
# =========================
@app.post("/summarize/")
def summarize(data: DialogueInput):
    result = summarize_text(data.dialogue)
    return {"summary": result}