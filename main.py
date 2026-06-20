from fastapi import FastAPI, File, UploadFile
app = FastAPI()
@app.get("/")
def read_root():
    return{"message": "ReadMyResults backend is alive"}

@app.post("/upload")
async def upload_documant(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents)
    }