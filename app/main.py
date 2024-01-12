import pathlib
import os
import io
import uuid
import uvicorn
from functools import lru_cache
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    File,
    UploadFile,
    Header
    )
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic_settings import BaseSettings
from .AIFunctions import extract_text_from_pdf, generate_text_summary, extract_names, extract_text_from_txt, ask_question

baseAddress = ""

class Settings(BaseSettings):
    debug: bool = False
    echo_active: bool = False
    app_auth_token: str = ""
    app_auth_token_prod: str = ""
    skip_auth: bool = False

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()

settings = get_settings()
DEBUG=settings.debug

BASE_DIR = pathlib.Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"

app = FastAPI()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def verify_auth(authorization = Header(None), settings:Settings = Depends(get_settings)):
    if settings.debug and settings.skip_auth:
        return
    if authorization is None:
        raise HTTPException(detail="Invalid authorization", status_code=401)
    label, token = authorization.split()
    if token != settings.app_auth_token:
        raise HTTPException(detail="Invalid authorization", status_code=401)

@app.get(baseAddress + "/", response_class=HTMLResponse)
def home_view(request: Request, settings:Settings = Depends(get_settings)):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post(baseAddress + "/")
async def file_analysis_view(file:UploadFile = File(...), authorization = Header(None), settings:Settings = Depends(get_settings)):
    verify_auth(authorization, settings)
    data = {}
    UPLOAD_DIR.mkdir(exist_ok=True)

    bytes_str = io.BytesIO(await file.read())
    fname = pathlib.Path(file.filename)
    fext = fname.suffix
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"

    with open(str(dest), 'wb') as out:
        out.write(bytes_str.read())

    file_extension = str(fname).split('.')[-1].lower()

    try:
        if file_extension == 'txt':
            text = extract_text_from_txt(dest)
            summary = generate_text_summary(text)
            names = extract_names(text)
            data = {
                "filetype":"Plain text document",
                "summary":summary,
                "names":names,
                "text":text
                }
        elif file_extension == 'doc' or file_extension == 'docx':
            text = extract_text_from_doc(dest)
            summary = generate_text_summary(text)
            names = extract_names(text)
            data = {
                "filetype":"Word document",
                "summary":summary,
                "names":names,
                "text":text
                }
        elif file_extension == 'pdf':
            text = extract_text_from_pdf(dest)
            summary = generate_text_summary(text)
            names = extract_names(text)
            data = {
                "filetype":"PDF document",
                "summary":summary,
                "names":names,
                "text":text
                }
        else:
            data = {"filetype":"Unknown"}

    # Delete the file from the uploads directory
    finally:
        try:
            dest.unlink()  # Delete the file
        except Exception as e:
            print(f"Error deleting file: {e}")

    return data

@app.post(baseAddress + "/chat/")
async def chat_view(requestData: dict = None, authorization = Header(None), settings:Settings = Depends(get_settings)):
    verify_auth(authorization, settings)
    data = {}

    question = requestData.get('question')
    context = requestData.get('context')

    data = ask_question(question, context)

    return data

@app.post(baseAddress + "/file-echo/", response_class=FileResponse)
async def file_upload(file:UploadFile = File(...), settings:Settings = Depends(get_settings)):
    if not settings.echo_active:
        raise HTTPException(detail="Invalid endpoint", status_code=400)

    UPLOAD_DIR.mkdir(exist_ok=True)

    bytes_str = io.BytesIO(await file.read())
    fname = pathlib.Path(file.filename)
    fext = fname.suffix
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"

    with open(str(dest), 'wb') as out:
        out.write(bytes_str.read())
    return dest