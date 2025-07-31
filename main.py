from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
import soundfile as sf
import io

app = FastAPI()
pipeline = KPipeline(lang_code='a')  # American English default

@app.post("/v1/audio/speech")
async def tts(input: str = Form(...), voice: str = Form("am_michael")):
    generator = pipeline(input, voice=voice)
    audio_buffer = io.BytesIO()

    for i, (_, _, audio) in enumerate(generator):
        sf.write(audio_buffer, audio, 24000, format='MP3')

    audio_buffer.seek(0)
    return StreamingResponse(audio_buffer, media_type="audio/mpeg")
