import base64
import time
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
from pathlib import Path
from typing import Optional
import uuid
import wave
import struct
import math
from elevenlabs.client import ElevenLabs
import requests
from elevenlabs import play


app = FastAPI(title="NOVEXA AGI TTS API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for audio files if it doesn't exist
TEMP_AUDIO_DIR = Path("audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Mount the static folder (frontend)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    """Serve the main HTML page"""
    return FileResponse("static/novexa_template.html")

@app.post("/tts")
async def generate_tts(
    provider: str = Form(...),
    api_key: str = Form(...),
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    """Generate text-to-speech audio"""
    
    # Validate input
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")
    
    # Validate provider
    supported_providers = ["elevenlabs", "google", "openai"]
    if provider not in supported_providers:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported provider. Supported providers: {', '.join(supported_providers)}"
        )
    
    try:
        # Generate unique filename
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        
        print(provider, api_key, text, voice, model, language)
        # For now, create dummy audio (replace this with actual TTS API calls)
        if provider == "elevenlabs":
            audio_path = await generate_elevenlabs_tts(text, api_key, voice, model, audio_filename)
        elif provider == "google":
            audio_path = await generate_google_tts(text, api_key, voice, language, audio_filename)
        elif provider == "openai":
            audio_path = await generate_openai_tts(text, api_key, voice, model, audio_filename)
        
        # Return the audio file URL and metadata
        return JSONResponse({
            "success": True,
            "audio_url": f"/audio/{audio_filename}",
            "audio_id": audio_id,
            "provider": provider,
            "text_length": len(text),
            "filename": audio_filename,
            "message": "Audio generated successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# @app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Serve generated audio files"""
    audio_path = TEMP_AUDIO_DIR / f"{audio_id}.wav"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"tts_audio_{audio_id}.wav",
        headers={"Cache-Control": "no-cache"}
    )

@app.post("/test-connection")
async def test_connection(
    provider: str = Form(...),
    api_key: str = Form(...)
):
    """Test API connection for the specified provider"""
    
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")
    
    try:
        # Simulate connection test (replace with actual API validation)
        if provider == "elevenlabs":
            success = await test_elevenlabs_connection(api_key)
        elif provider == "google":
            success = await test_google_connection(api_key)
        elif provider == "openai":
            success = await test_openai_connection(api_key)
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")
        
        if success:
            return JSONResponse({
                "success": True,
                "message": f"{provider.title()} connection successful",
                "provider": provider
            })
        else:
            return JSONResponse({
                "success": False,
                "message": f"{provider.title()} connection failed",
                "provider": provider
            }, status_code=401)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

# TTS Provider Functions (replace these with actual API integrations)

async def generate_elevenlabs_tts(text: str, api_key: str, voice: str, model: str, filename: str) -> str:
    """Generate TTS using ElevenLabs API"""
    elevenlabs = ElevenLabs(
    api_key=api_key,
    )
    # fallback defaults
    if not voice:
        voice = "JBFqnCBsd6RMkjVDRZzb"
    if not model:
        model = "eleven_multilingual_v2"

    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id=voice,
        model_id=model,
        output_format="mp3_44100_128",
    )
    print(f"ElevenLabs TTS: {text[:50]}... (Voice: {voice}, Model: {model}, api_key: {api_key})")
    file_path = os.path.join("audio", filename) 
    os.makedirs("audio", exist_ok=True)
    with open(file_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return file_path

async def generate_google_tts(text: str, api_key: str, voice: str, language: str, filename: str) -> str:
    """Generate TTS using Google Cloud Text-to-Speech REST API."""

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": language or "en-US",
            "name": voice or "en-US-Wavenet-D"
        },
        "audioConfig": {"audioEncoding": "LINEAR16"}
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    audio_base64 = result.get("audioContent")
    if not audio_base64:
        raise Exception(f"Google TTS error: {result}")

    audio_data = base64.b64decode(audio_base64)

    os.makedirs("audio", exist_ok=True)
    file_path = os.path.join("audio", filename)

    with open(file_path, "wb") as f:
        f.write(audio_data)

    print(f"Google TTS saved to {file_path}")
    return file_path

async def generate_openai_tts(text: str, api_key: str, voice: str, model: str, filename: str) -> str:
    """Generate TTS using OpenAI TTS API"""
    pass

# Connection Test Functions (replace with actual API validation)

async def test_elevenlabs_connection(api_key: str) -> bool:
    """Test ElevenLabs API key by fetching voices"""
    try:
        client = ElevenLabs(api_key=api_key)

        # Tiny test: just one word
        audio_stream = client.text_to_speech.convert(
            text="Hi",
            voice_id="JBFqnCBsd6RMkjVDRZzb",   # default voice
            model_id="eleven_multilingual_v2", # safe default model
            output_format="mp3_44100_128",
        )

        # Force the generator to fetch data (trigger API call)
        first_chunk = next(audio_stream, None)
        # If no exception is raised, API key works ✅
        return first_chunk is not None  
    except Exception as e:
        print(f"❌ ElevenLabs test failed: {str(e)}")
        return False

async def test_google_connection(api_key: str) -> bool:
    """Test Google Cloud TTS API key by listing voices."""
    try:
        url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            voices = response.json().get("voices", [])
            print(f"✅ Google TTS key valid. Found {len(voices)} voices.")
            return True
        else:
            print(f"❌ Google TTS key invalid - {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Google TTS connection error: {str(e)}")
        return False

async def test_openai_connection(api_key: str) -> bool:
    """Test OpenAI API connection"""
    pass


@app.delete("/audio/{audio_id}")
async def delete_audio(audio_id: str):
    """Delete generated audio file"""
    audio_path = TEMP_AUDIO_DIR / f"{audio_id}.wav"
    
    if audio_path.exists():
        os.remove(audio_path)
        return {"success": True, "message": "Audio file deleted"}
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# Cleanup old audio files on startup
@app.on_event("startup")
async def cleanup_old_files():
    """Clean up old audio files on server startup"""
    try:
        now = time.time()
        cutoff = now - 3600

        for file_path in TEMP_AUDIO_DIR.glob("*.wav"):
            # Remove files older than 1 hour
            if file_path.stat().st_mtime < cutoff:
                os.remove(file_path)
                print(f"🗑️ Deleted old file: {file_path.name}")
        print(f"Cleaned up old audio files in {TEMP_AUDIO_DIR}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)