from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import io
import asyncio
import logging
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kokoro TTS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pipeline = KPipeline(lang_code='a')
    logger.info("Kokoro pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Kokoro pipeline: {e}")
    pipeline = None

VALID_VOICES = [
    "am_adam",      # deep documentary narrator — recommended for SandsOfTime
    "am_michael",
    "am_emma",
    "am_olivia",
    "am_sophia",
    "am_isabella",
]
MAX_INPUT_CHARS = 50000   # ~10,000 words — covers a full 20-min documentary script
CHUNK_SIZE = 450           # safe quality ceiling per Kokoro call
SAMPLE_RATE = 24000


def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks at sentence boundaries so no chunk exceeds
    max_chars. Falls back to comma-boundary, then word-boundary splitting
    for pathologically long sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)

            if len(sentence) <= max_chars:
                current = sentence
            else:
                # Long sentence: split at commas first
                parts = re.split(r'(?<=,)\s+', sentence)
                current = ""
                for part in parts:
                    if len(current) + len(part) + 1 <= max_chars:
                        current = (current + " " + part).strip()
                    else:
                        if current:
                            chunks.append(current)
                        if len(part) <= max_chars:
                            current = part
                        else:
                            # Last resort: split by words
                            words = part.split()
                            current = ""
                            for word in words:
                                if len(current) + len(word) + 1 <= max_chars:
                                    current = (current + " " + word).strip()
                                else:
                                    if current:
                                        chunks.append(current)
                                    current = word

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


@app.get("/health")
async def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable")
    return {"status": "healthy", "service": "kokoro-tts", "version": "2.0.0"}


@app.get("/")
async def root():
    return {
        "message": "Kokoro TTS API",
        "version": "2.0.0",
        "max_input_chars": MAX_INPUT_CHARS,
        "voices": VALID_VOICES,
        "default_voice": "am_adam",
    }


@app.get("/voices")
async def list_voices():
    return {"voices": VALID_VOICES, "default": "am_adam"}


@app.post("/v1/audio/speech")
async def tts(
    request: Request,
    input: str = Form(...),
    voice: str = Form("am_adam"),
):
    """
    Generate speech from text. Handles full documentary scripts via
    automatic sentence-boundary chunking and audio concatenation.

    Returns a single WAV file regardless of input length.
    """
    start_time = time.time()

    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable")

    if not input or not input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    if len(input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long: {len(input)} chars (max {MAX_INPUT_CHARS})",
        )

    if voice not in VALID_VOICES:
        logger.warning(f"Unknown voice '{voice}', using am_adam")
        voice = "am_adam"

    chunks = chunk_text(input)
    logger.info(
        f"TTS start: voice={voice}, total_chars={len(input)}, chunks={len(chunks)}"
    )

    async def generate_audio():
        all_audio: list[np.ndarray] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i + 1}/{len(chunks)}: {len(chunk)} chars")
            try:
                generator = pipeline(chunk, voice=voice)
                for _, _, audio in generator:
                    if audio is not None and len(audio) > 0:
                        all_audio.append(audio)
            except Exception as exc:
                logger.error(f"Chunk {i + 1} failed: {exc}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio generation failed on chunk {i + 1}: {exc}",
                )

        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio was generated")

        combined = np.concatenate(all_audio)
        buffer = io.BytesIO()
        sf.write(buffer, combined, SAMPLE_RATE, format="WAV")
        buffer.seek(0)
        return buffer

    # Dynamic timeout: 35 seconds per chunk plus a 15-second buffer
    timeout_seconds = len(chunks) * 35 + 15

    try:
        audio_buffer = await asyncio.wait_for(generate_audio(), timeout=timeout_seconds)
        processing_time = round(time.time() - start_time, 2)
        logger.info(
            f"TTS done: {processing_time}s | chunks={len(chunks)} | "
            f"chars={len(input)}"
        )

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Processing-Time": str(processing_time),
                "X-Chunks-Processed": str(len(chunks)),
                "X-Input-Chars": str(len(input)),
            },
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {timeout_seconds}s",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, workers=1, timeout_keep_alive=120)
