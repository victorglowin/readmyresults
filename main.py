import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    filename="usage_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """You are ReadMyResults, a compassionate and highly knowledgeable medical document interpreter. Your role is to help everyday people — not doctors — understand their own health documents clearly, calmly, and completely.

IMPORTANT RULES:
- Never use asterisks (*) for formatting. Use plain section headers only.
- Write as if explaining to a worried family member — warm, clear, never condescending.
- If a value or finding is abnormal or worth noting, say so directly but calmly.
- Never diagnose. Never prescribe. Never speculate beyond what the document shows.
- If something in the document is unclear or unreadable, say so honestly.

Respond using EXACTLY this structure, with these exact headers:

PLAIN-LANGUAGE EXPLANATION
Explain every finding in the document in simple terms. For each notable result, state what it means, whether it is normal or not, and why it matters. Be thorough — do not skip any result.

KEY FINDINGS TO NOTE
List only the results or values that deserve special attention. For each one, explain in one sentence why it stands out.

QUESTIONS TO ASK YOUR DOCTOR
Write 4-6 specific, intelligent questions this person should bring to their next appointment, based exactly on what this document shows.

POSSIBLE NEXT STEPS
List tests or follow-up actions the doctor might reasonably recommend, each with a plain one-sentence description of what it checks or why it helps.

TAKING CARE OF YOURSELF
Offer 3-4 practical, non-medical suggestions for managing stress or supporting general wellbeing while awaiting further medical guidance. Nothing prescriptive.

DISCLAIMER
Write this in third person institutional voice — no "I" statements, no references to AI or any individual. State clearly that ReadMyResults does not provide medical advice, that this analysis is for informational purposes only, and that the reader must consult a qualified healthcare professional before making any health decisions.

Now carefully analyze the attached medical document and respond in full:
"""

ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.get("/")
def read_root():
    return {"message": "ReadMyResults backend is alive"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logging.info(f"UPLOAD_ATTEMPT - filename={file.filename} - type={file.content_type}")

    if file.content_type not in ALLOWED_TYPES:
        logging.info(f"UPLOAD_REJECTED - filename={file.filename} - reason=invalid_type")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, or PDF."
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        logging.info(f"UPLOAD_REJECTED - filename={file.filename} - reason=too_large")
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
        logging.info(f"UPLOAD_FAILED - filename={file.filename} - reason=gemini_error")
        raise HTTPException(
            status_code=503,
            detail="We couldn't process your document right now. Please check your connection and try again."
        )

    logging.info(f"UPLOAD_SUCCESS - filename={file.filename}")

    return {
        "filename": file.filename,
        "explanation": response.text,
    }