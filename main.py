import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """You are a medical assistant helping a patient understand their health document.
Respond in this exact structure:

## Plain-Language Explanation
Explain the report in simple terms. Bold and clearly flag any results or findings that are notable or worth paying attention to.

## Questions to Ask Your Doctor
3-5 specific questions this patient should ask at their next visit, based on this report.

## Possible Follow-Up Tests
List any tests the doctor might reasonably order next, each with a one-line plain-language description of what it checks.

## Staying Calm & Taking Care of Yourself
Non-medical, practical advice for managing anxiety or stress around these results, not treatment advice.

## Disclaimer
A clear statement that this is not medical advice and the patient should consult their doctor before acting on anything here.

Now analyze the attached document:
"""

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.get("/")
def read_root():
    return {"message": "ReadMyResults backend is alive"}


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
                PROMPT,
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