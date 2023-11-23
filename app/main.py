import pathlib
import os
import io
import uuid
import PyPDF2
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

class Settings(BaseSettings):
    debug: bool = False
    echo_active: bool = False
    app_auth_token: str
    app_auth_token_prod: str = None
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


@app.get("/", response_class=HTMLResponse)
def home_view(request: Request, settings:Settings = Depends(get_settings)):
    print(DEBUG)
    return templates.TemplateResponse("home.html", {"request": request})

def verify_auth(authorization = Header(None), settings:Settings = Depends(get_settings)):
    if settings.debug and settings.skip_auth:
        return
    if authorization is None:
        raise HTTPException(detail="Invalid authorization", status_code=401)
    label, token = authorization.split()
    if token != settings.app_auth_token:
        raise HTTPException(detail="Invalid authorization", status_code=401)


@app.post("/")
def file_analysis_view(file:UploadFile = File(...), authorization = Header(None), settings:Settings = Depends(get_settings)):
    verify_auth(authorization, settings)
    return {"hello":"world"}


@app.post("/file-echo/", response_class=FileResponse)
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