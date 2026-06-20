import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def read_root():
    return {"message": "ReadMyResults backend is alive"}

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, or PDF."
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Please upload a file under 10MB."
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "You are a medical assistant. Explain this health document in simple, "
                "plain language a non-medical person can understand. Be clear and reassuring.",
                types.Part.from_bytes(data=contents, mime_type=file.content_type),
            ],
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="We couldn't process your document right now. Please check your connection and try again."
        )

    return {
        "filename": file.filename,
        "explanation": response.text,
    }