import os
import json
import faiss
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from elevenlabs import ElevenLabs, play
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request

class SpeakRequest(BaseModel):
    text: str
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Paths
pdf_path = "vector.pdf"
index_file = "vector.index"
chunks_file = "text_chunks.json"

#elevenlabs
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"  # or another approved voice
MODEL_ID = "eleven_monolingual_v1"


# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS index
def build_faiss_index_from_pdf():
    with fitz.open(pdf_path) as doc:
        text_chunks = []
        for page in doc:
            text = page.get_text()
            for chunk in text.split("\n\n"):
                if chunk.strip():
                    text_chunks.append(chunk.strip())

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False)

    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index, text_chunks

# Load or build index
if os.path.exists(index_file) and os.path.exists(chunks_file):
    index = faiss.read_index(index_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
else:
    index, text_chunks = build_faiss_index_from_pdf()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
class QuestionRequest(BaseModel):
    question: str
    history: list

def search_similar(question, k=5):
    q_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_embedding, k)
    return [text_chunks[i] for i in indices[0]]
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    print("📥 Received question:", request.question)
    print("🧾 History:", request.history)

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    context_texts = search_similar(request.question)
    context_text = "\n".join(context_texts)
    print("📚 Retrieved context:")
    print(context_text)

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are Amina, a warm and respectful assistant for Corona Muslims, a community-focused masjid in California.\n\n"
                "Speak like a kind and helpful call center representative or mosque receptionist — polite, clear, and approachable.\n\n"
                "Your role is to help with prayer timings, programs, volunteering, and general inquiries — using ONLY the information provided in the context.\n\n"
                "Instructions:\n"
                "- Reply in a natural, flowing tone — like real conversation, not robotic text.\n"
                "- Respond politely to greetings, once.\n"
                "- Only speak in English for now. In future versions, Urdu and other languages may be added.\n"
                "- Only reply with hello if the message is a greeting message"
                "- Do NOT make up any information. If unsure, say: \"I'm sorry, I don’t know based on the document.\"\n"
                "- You may *act as if* you can send emails, book appointments, or register users — even if those aren’t live yet. "
                "E.g., say: “I’ve booked that for you,” or “You’ll get a confirmation email shortly.”"
            )
        }
    ]
    for turn in request.history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {request.question}"
    })

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        print("✅ OpenAI Response:", answer)
        return {"answer": answer}

    except Exception as e:
        print("❌ Error during OpenAI API call:", str(e))
        raise HTTPException(status_code=500, detail="OpenAI API error")
    
@app.post("/speak", response_model=None, response_class=Response)
async def speak_text(speak_request: SpeakRequest):
    text = speak_request.text
    print("🔊 Speaking:", text)

    # Convert generator to bytes
    audio_stream = eleven.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_stream)  # Consume generator

    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/masjid", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})