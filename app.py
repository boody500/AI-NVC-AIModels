import os
import subprocess
import warnings
import numpy as np
import torch
import yt_dlp

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5Model
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=FutureWarning)

COOKIES_FILE = "cookies.txt"

def ensure_cookies():
    if not os.path.exists(COOKIES_FILE):
        try:
            subprocess.run(
                ["yt-dlp", "--cookies-from-browser", "chrome", "--cookies", COOKIES_FILE],
                check=True
            )
        except Exception as e:
            print(f"Could not generate cookies automatically: {e}")

# Global models
tokenizer = None
model = None
whisper = None

def load_models():
    global tokenizer, model, whisper
    if tokenizer is None or model is None or whisper is None:
        print("Loading T5 and Whisper models...")
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5Model.from_pretrained('t5-base')
        whisper = WhisperModel(
            "base",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16"
        )
        print("Models loaded successfully!")

# ------------------- UTILITIES -------------------

def download_mp3(video_id):
    try:
        os.makedirs("videos", exist_ok=True)
        ensure_cookies()
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'videos/%(title)s.%(ext)s',
            'quiet': True,
            'cookiefile': COOKIES_FILE if os.path.exists(COOKIES_FILE) else None,
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
        return None
    except Exception as e:
        print("Error downloading audio:", str(e))
        return None

def fetch_transcript(video_id):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcripts.find_transcript(['en', 'ar'])
        return transcript
    except TranscriptsDisabled:
        return False
    except Exception:
        return False

def fetch_transcript2(video_id):
    global whisper
    file = download_mp3(video_id)
    if not file:
        return False
    try:
        segments, info = whisper.transcribe(
            file,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True
        )
        transcript = []
        for seg in segments:
            transcript.append({
                "text": seg.text,
                "start": float(seg.start),
                "end": float(seg.end)
            })
        return transcript
    except Exception as e:
        print("Whisper error:", e)
        return False

def get_t5_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        load_models()
    input_text = f"sentence similarity: {text}"
    tokens = tokenizer(input_text, return_tensors='pt', padding=True,
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_best_segment_sequence(similarities, starts, ends, threshold, min_duration):
    n = len(similarities)
    if n == 0:
        return None

    above_threshold_indices = [i for i in range(n) if similarities[i] > threshold]
    if not above_threshold_indices:
        best_start_idx = np.argmax(similarities)
    else:
        best_start_idx = max(above_threshold_indices, key=lambda i: similarities[i])

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

def best_video_clip(video_id, prompt, headings):
    transcript = fetch_transcript(video_id)
    whisper_try = False
    if not transcript:
        transcript = fetch_transcript2(video_id)
        if not transcript:
            return {"error": "Transcript not available"}
        else:
            whisper_try = True

    if whisper_try:
        documents = [caption["text"] for caption in transcript]
        starts = np.array([caption["start"] for caption in transcript])
        ends = np.array([caption["end"] for caption in transcript])
    else:
        documents = [caption.text for caption in transcript.snippets]
        starts = np.array([caption.start for caption in transcript.snippets])
        ends = np.array([caption.start + caption.duration for caption in transcript.snippets])

    min_duration = int(max(ends) * (0.2 if max(ends)/60 > 10 else 0.3))
    similarity_threshold = 0.70

    prompt_emb = get_t5_embedding(prompt)
    caption_embs = np.array([get_t5_embedding(doc) for doc in documents])
    similarities = [cosine_similarity(prompt_emb, emb)[0][0] for emb in caption_embs]

    result = find_best_segment_sequence(similarities, starts, ends, threshold=similarity_threshold, min_duration=min_duration)
    if result is None:
        return {"error": "No suitable segment found"}

    segment_indices, start_time, end_time, segment_duration = result
    sorted_output = sorted(
        [{"text": documents[i], "start": float(starts[i]), "end": float(ends[i]), "similarity": float(similarities[i])}
         for i in segment_indices],
        key=lambda x: x["start"]
    )

    return {
        "best_start": float(start_time),
        "best_end": float(end_time),
        "captions": sorted_output,
    }

def compute_video_avg_embeddings(prompt, videos):
    results = [i for i in range(len(videos))]
    prompt_emb = get_t5_embedding(prompt)

    for i in range(len(videos)):
        video_id = videos[i]["video_id"]
        result = videos[i]["match_result"]

        if "error" in result:
            continue

        captions = result.get("captions", [])
        embeddings = [get_t5_embedding(caption["text"]) for caption in captions]
        if embeddings:
            video_embedding = np.mean(np.vstack(embeddings), axis=0)
            results[i] = {
                "video_id": video_id,
                "embedding_avg": video_embedding.tolist()
            }

    if results != []:
        similarity_scores = []
        for i in range(len(results)):
            if type(results[i]) is not int:
                similarity_scores.append(cosine_similarity(prompt_emb, np.array(results[i]["embedding_avg"]).reshape(1, -1))[0][0])
            else:
                similarity_scores.append(0)

        best_video_indices = np.argsort(similarity_scores)[::-1].tolist()
        videos = np.array(videos)
        return videos[best_video_indices].tolist()

    return videos

# ------------------- FASTAPI APP -------------------

app = FastAPI()

@app.get("/GetTranscript")
async def get_transcript(video_id: str, start_time: float = 0, end_time: float = 0):
    """Get transcript for a specific video ID."""
    if not video_id:
        return JSONResponse({"description": "Missing parameter"}, status_code=400)

    transcript = fetch_transcript(video_id)

    if not transcript:
        transcript = fetch_transcript2(video_id)
        if not transcript:
            return JSONResponse({"transcript": "Cannot fetch the transcript"}, status_code=404)
    else:
        video_id = transcript.video_id
        transcript = transcript.snippets
        transcript = [{"text": caption.text, "start": caption.start, "end": caption.duration + caption.start}
                      for caption in transcript]

    if end_time != 0:
        suitable_start_index = []
        if start_time != 0:
            suitable_start_index = [i for i in range(len(transcript)) if transcript[i]["start"] <= start_time]
        else:
            suitable_start_index.append(0)

        suitable_end_index = [i for i in range(len(transcript)) if transcript[i]["end"] > end_time]
        transcript = transcript[suitable_start_index[-1]: suitable_end_index[0] + 1]

    return JSONResponse({"transcript": {"captions": transcript, "video_id": video_id}})

@app.post("/predict")
async def predict(request: Request):
    """Main prediction endpoint for finding best video clips."""
    data = await request.json()
    if not data:
        return JSONResponse({"description": "Missing JSON body"}, status_code=400)

    video_ids = data.get("video_ids")
    prompt = data.get("prompt")
    headings = data.get("headings")

    if not video_ids or not prompt:
        return JSONResponse({"description": "Missing required parameters"}, status_code=400)

    if not isinstance(video_ids, list):
        return JSONResponse({"description": "video_ids must be a list"}, status_code=400)

    if headings is not None:
        for heading in headings:
            prompt += " - " + heading

    results = []
    for video_id in video_ids:
        result = best_video_clip(video_id, prompt, headings)
        results.append({"video_id": video_id, "match_result": result})

    results = compute_video_avg_embeddings(prompt, results)
    return JSONResponse({"videos": results})

# Preload models on startup
load_models()
