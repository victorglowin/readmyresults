import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def read_root():
    return {"message": "ReadMyResults backend is alive"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "You are a medical assistant. Explain this health document in simple, "
            "plain language a non-medical person can understand. Be clear and reassuring.",
            types.Part.from_bytes(data=contents, mime_type=file.content_type),
        ],
    )

    return {
        "filename": file.filename,
        "explanation": response.text,
    }