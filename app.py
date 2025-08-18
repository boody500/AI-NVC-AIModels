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
def fetch_transcript(video_id, max_retries=5, initial_delay=1):
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
        # Clean up the file after processing
        if os.path.exists(file):
            os.remove(file)
        return [{"text": s.text, "start": float(s.start), "end": float(s.end)} for s in segments]
    except:
        return False

# ------------------- EMBEDDING -------------------

def get_t5_embedding(text):
    """Get T5 embeddings for a given text."""
    global tokenizer, model
    if tokenizer is None or model is None:
        load_models()

    input_text = f"sentence similarity: {text}"
    tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_best_segment_sequence(similarities, starts, ends, threshold, min_duration):
    """Find the best segment sequence based on similarity scores."""
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
    """Find the best video clip segment for a given prompt."""
    transcript = fetch_transcript(video_id)
    whisper_try = False
    if not transcript:
        print("try to run whisper")
        transcript = fetch_transcript2(video_id)
        if not transcript :
          return {"error": "Transcript not available"}
        else:
          whisper_try = True
          print("whisper done!")

    documents = []
    starts = []
    ends = []
    if whisper_try:
        documents = [caption["text"] for caption in transcript]
        starts = np.array([caption["start"] for caption in transcript])
        ends = np.array([caption["end"] for caption in transcript])
    else:
        documents = [caption.text for caption in transcript.snippets]
        starts = np.array([caption.start for caption in transcript.snippets])
        ends = np.array([caption.start + caption.duration for caption in transcript.snippets])

    min_duration = 0
    if max(ends)/60 > 10:
        min_duration = int(max(ends) * 0.2)
    else:
        min_duration = int(max(ends) * 0.3)

    similarity_threshold = 0.70

    prompt_emb = get_t5_embedding(prompt)
    caption_embs = np.array([get_t5_embedding(doc) for doc in documents])
    similarities = [cosine_similarity(prompt_emb, emb)[0][0] for emb in caption_embs]

    result = find_best_segment_sequence(similarities, starts, ends,
                                      threshold=similarity_threshold,
                                      min_duration=min_duration)
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
    """Compute average T5 embedding for each valid video."""
    results = [i for i in range(len(videos))]
    prompt_emb = get_t5_embedding(prompt)

    for i in range(len(videos)):
        video_id = videos[i]["video_id"]
        result = videos[i]["match_result"]

        if "error" in result:
            print(f"Skipping video {video_id}: {result['error']}")
            continue

        captions = result.get("captions", [])
        embeddings = []

        for caption in captions:
            text = caption["text"]
            emb = get_t5_embedding(text)
            embeddings.append(emb)

        if embeddings:
            video_embedding = np.mean(np.vstack(embeddings), axis=0)
            results[i] = {
                "video_id": video_id,
                "embedding_avg": video_embedding.tolist()  # convert to list for JSON compatibility
            }

    if results != []:
        similarity_scores = []
        for i in range(len(results)):
            if type(results[i]) is not int:
                similarity_scores.append(cosine_similarity(prompt_emb, np.array(results[i]["embedding_avg"]).reshape(1, -1))[0][0])
            else:
                similarity_scores.append(0)

        print(f"Similarity scores: {similarity_scores}")
        best_video_indices = np.argsort(similarity_scores)[::-1].tolist()
        print(f"Best video indices: {best_video_indices}")
        videos = np.array(videos)
        return videos[best_video_indices].tolist()

    return videos

def ensure_models_loaded():
    if tokenizer is None or model is None or whisper is None:
        load_models()


# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return jsonify({"message": "YouTube Transcript Analysis API is running!", "version": "1.0.0"})

@app.route("/GetTranscript", methods=["GET"])
def get_transcript():
    """Get transcript for a specific video ID."""
    ensure_models_loaded()  # <--- Ensure models are ready

    video_id = request.args.get("video_id")
    start_time = float(request.args.get("start_time"))
    end_time = float(request.args.get("end_time"))
    print(video_id)
    print(start_time)
    print(end_time)
    if not video_id  :
        return jsonify({"description": "Missing parameter"}), 400

    transcript = fetch_transcript(video_id)
    print(transcript)

    if not transcript:
      print("try to run whisper")
      transcript = fetch_transcript2(video_id)
      if not transcript :
        return jsonify({"transcript": "Cannot fetch the transcript"}), 404
      else:
        print("wisper runned successfully!")


    else:
      #for function 1 as it doesn't return end time
      video_id = transcript.video_id
      transcript = transcript.snippets
      transcript = [{"text": caption.text, "start": caption.start, "end": caption.duration + caption.start} for caption in transcript]

    if end_time != 0:
      suitable_start_index = []
      if start_time != 0:
        suitable_start_index = [i for i in range(len(transcript)) if transcript[i]["start"] <= start_time]
      else:
        suitable_start_index.append(0)

      suitable_end_index = [i for i in range(len(transcript)) if transcript[i]["end"] > end_time]
      print(suitable_start_index)
      print(suitable_end_index)
      transcript = transcript[suitable_start_index[len(suitable_start_index)-1] : suitable_end_index[0]+1]
      print(transcript)

    return jsonify({"transcript":{"captions" : transcript,"video_id":video_id}}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint for finding best video clips."""
    ensure_models_loaded()  # <--- Ensure models are ready

    data = request.get_json()
    if not data:
        return jsonify({"description": "Missing JSON body"}), 400

    video_ids = data.get("video_ids")
    prompt = data.get("prompt")
    headings = data.get("headings")

    if not video_ids or not prompt:
        return jsonify({"description": "Missing required parameters"}), 400

    if not isinstance(video_ids, list):
        return jsonify({"description": "video_ids must be a list"}), 400

    if headings is not None:
        for heading in headings:
            prompt += " - " + heading
        print(f"Using headings in prompt: {headings}")
        print(f"Updated prompt: {prompt}")

    results = []
    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        result = best_video_clip(video_id, prompt, headings)
        results.append({"video_id": video_id, "match_result": result})

    results = compute_video_avg_embeddings(prompt, results)
    return jsonify({"videos": results})


# ------------------- MAIN ENTRY -------------------


def run_app():
    port = int(os.environ.get("PORT", 8080))  # Azure uses dynamic port
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    run_app()
