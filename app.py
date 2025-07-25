import json
import numpy as np
import torch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import warnings
import requests
import os
import yt_dlp
import webvtt
from faster_whisper import WhisperModel

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

# ------------------- GPU/CPU FALLBACK -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

tokenizer = None
model = None
whisper = None

def load_models():
    global tokenizer, model, whisper
    if tokenizer is None or model is None or whisper is None:
        print("Loading T5 model...")
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5Model.from_pretrained('t5-base')
        print(f"Loading Whisper on {DEVICE}...")
        whisper = WhisperModel("base", device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        print("Models loaded successfully!")

# ------------------- COOKIE FUNCTION -------------------
def get_youtube_cookies():
    try:
        session = requests.Session()
        session.get("https://www.youtube.com", timeout=10)
        cookie_path = './cookies.txt'
        with open(cookie_path, 'w') as f:
            f.write("# Netscape HTTP Cookie File\n")
            for cookie in session.cookies:
                domain = cookie.domain
                secure = "TRUE" if cookie.secure else "FALSE"
                path = cookie.path
                expiry = str(int(cookie.expires)) if cookie.expires else "0"
                name = cookie.name
                value = cookie.value
                f.write(f"{domain}\tTRUE\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
        print(f"✅ Real-time cookies saved to {cookie_path}")
        return cookie_path
    except Exception as e:
        print(f"❌ Failed to generate cookies: {e}")
        return None

# ------------------- TRANSCRIPT FETCH -------------------
def fetch_transcript(video_id, max_retries=5, initial_delay=5):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcripts.find_transcript(['en', 'ar'])
            return transcript.fetch()
        except TranscriptsDisabled:
            return False
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return False

def download_mp3(video_id):
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        cookie_file = get_youtube_cookies()

        if not os.path.exists("videos"):
            os.makedirs("videos")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'videos/%(title)s.%(ext)s',
            'quiet': False,
            'cookiefile': cookie_file if cookie_file else None,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        for file in os.listdir("videos"):
            if file.endswith('.mp3'):
                return os.path.join("videos", file)
        return False
    except Exception as e:
        print("Error:", str(e))
        return False

def fetch_transcript2(video_id):
    file = download_mp3(video_id)
    if not file:
        return False
    try:
        segments, info = whisper.transcribe(
            file, beam_size=5, vad_filter=True, word_timestamps=True
        )
        return [{"text": s.text, "start": float(s.start), "end": float(s.end)} for s in segments]
    except:
        return False

# ------------------- EMBEDDING -------------------
def get_t5_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        load_models()
    input_text = f"sentence similarity: {text}"
    tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_best_segment_sequence(similarities, starts, ends, threshold, min_duration):
    n = len(similarities)
    if n == 0:
        return None
    above_threshold_indices = [i for i in range(n) if similarities[i] > threshold]
    best_start_idx = max(above_threshold_indices, key=lambda i: similarities[i]) if above_threshold_indices else np.argmax(similarities)
    segment_indices = [best_start_idx]
    segment_start = starts[best_start_idx]
    segment_end = ends[best_start_idx]
    while segment_end - segment_start < min_duration:
        remaining_indices = [i for i in range(n) if i not in segment_indices]
        if not remaining_indices:
            break
        next_candidates = [i for i in remaining_indices if starts[i] >= segment_end]
        if not next_candidates:
            break
        next_idx = min(next_candidates, key=lambda i: starts[i])
        segment_indices.append(next_idx)
        segment_end = max(segment_end, ends[next_idx])
        segment_start = min(segment_start, starts[next_idx])
    segment_duration = segment_end - segment_start
    if segment_duration <= min_duration:
        return None
    segment_indices = sorted(segment_indices, key=lambda i: starts[i])
    return segment_indices, segment_start, segment_end, segment_duration

# ------------------- MAIN LOGIC -------------------
def best_video_clip(video_id, prompt, headings):
    transcript = fetch_transcript(video_id)
    whisper_try = False
    if not transcript:
        transcript = fetch_transcript2(video_id)
        if not transcript:
            return {"error": "Transcript not available"}
        whisper_try = True

    if whisper_try:
        documents = [c["text"] for c in transcript]
        starts = np.array([c["start"] for c in transcript])
        ends = np.array([c["end"] for c in transcript])
    else:
        documents = [c.text for c in transcript.snippets]
        starts = np.array([c.start for c in transcript.snippets])
        ends = np.array([c.start + c.duration for c in transcript.snippets])

    min_duration = int(max(ends) * 0.2 if max(ends)/60 > 10 else max(ends) * 0.3)
    prompt_emb = get_t5_embedding(prompt)
    caption_embs = np.array([get_t5_embedding(doc) for doc in documents])
    similarities = [cosine_similarity(prompt_emb, emb)[0][0] for emb in caption_embs]
    result = find_best_segment_sequence(similarities, starts, ends, threshold=0.7, min_duration=min_duration)
    if result is None:
        return {"error": "No suitable segment found"}
    segment_indices, start_time, end_time, _ = result
    sorted_output = sorted(
        [{"text": documents[i], "start": float(starts[i]), "end": float(ends[i]), "similarity": float(similarities[i])} for i in segment_indices],
        key=lambda x: x["start"]
    )
    return {"best_start": float(start_time), "best_end": float(end_time), "captions": sorted_output}

# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return jsonify({"message": "YouTube Transcript Analysis API is running!", "version": "1.0.0"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"description": "Missing JSON body"}), 400
    video_ids = data.get("video_ids")
    prompt = data.get("prompt")
    if not video_ids or not prompt:
        return jsonify({"description": "Missing parameters"}), 400
    results = []
    for vid in video_ids:
        results.append({"video_id": vid, "match_result": best_video_clip(vid, prompt, None)})
    return jsonify({"videos": results})

# ------------------- MAIN ENTRY -------------------
def run_app():
    port = int(os.environ.get("PORT", 5000))  # Azure uses dynamic port
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    load_models()
    run_app()
