import secrets
import time
import io
import wave
from flask import Flask, request, Response, jsonify
from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer, logging
from PIL import Image
from ultralytics import YOLO
from faster_whisper import WhisperModel
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment

# Configuration
lang_set = "ml-gtts"
gain_value = 1.0  # Gain control in dB

# Initialize devices and models
device, dtype = detect_device()
logging.set_verbosity_error()

listener_id = "distil-large-v3"
vision_model_id = "vikhyatk/moondream2"
tts_voice_id = "en_US-ljspeech-medium.onnx"

tokenizer = AutoTokenizer.from_pretrained(vision_model_id, revision=LATEST_REVISION)
ind_currency_model = YOLO("ind_currency.pt")
vision = Moondream.from_pretrained(
    vision_model_id, revision=LATEST_REVISION, torch_dtype=dtype, attn_implementation="flash_attention_2"
).to('cuda')
vision.eval()
listen = WhisperModel(listener_id, device="cuda", compute_type="float16")

currency_map = {
    0: "Ten Rupees",
    1: "Twenty Rupees",
    2: "Fifty Rupees",
    3: "Hundred Rupees",
    4: "Two Hundred Rupees",
    5: "Five Hundred Rupees",
    6: "Two Thousand Rupees"
}

app = Flask(__name__)

# Function to convert text to WAV using gTTS and pydub
def get_malayalam_wav(text):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='ml').text
    tts = gTTS(translated_text, lang='ml')
    audio_data_mp3 = io.BytesIO()
    tts.write_to_fp(audio_data_mp3)
    audio_data_mp3.seek(0)
    audio_segment = AudioSegment.from_mp3(audio_data_mp3)
    audio_segment = audio_segment.set_frame_rate(22050).set_sample_width(2)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(audio_segment.raw_data)
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

# Function to apply gain control to WAV data
def apply_gain(wav_data):
    audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))
    adjusted_segment = audio_segment + gain_value  # Apply gain
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(audio_segment.channels)
        wav_file.setsampwidth(audio_segment.sample_width)
        wav_file.setframerate(audio_segment.frame_rate)
        wav_file.writeframes(adjusted_segment.raw_data)
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

# Function to process request data and generate response
def process_request(data):
    start_time = time.time()
    try:
        parts = data.split(b'|*|')
        image_data, audio_data, user_id, action = parts[:4]

        print(f"UserID: {user_id.decode()}")
        print(f"Action: {action.decode()}")

        if action.decode() == 'A':
            prompt = "Describe this image."
            result = vision.answer_question(vision.encode_image(Image.open(io.BytesIO(image_data))), prompt, tokenizer)
        elif action.decode() == 'B':
            prompt = transcribe(io.BytesIO(audio_data), "en")
            result = vision.answer_question(vision.encode_image(Image.open(io.BytesIO(image_data))), prompt, tokenizer)
        elif action.decode() == 'C':
            result = ind_currency_detect(image_data)
        else:
            print("Invalid Action")
            return None

        if lang_set == "ml-gtts":
            wav_data = get_malayalam_wav(result)
        else:
            wav_data = get_malayalam_wav(result)  # Default to Malayalam for simplicity

        print(f"Execution_Time: {time.time() - start_time} seconds")
        print("Result: " + result)

        # Apply gain control
        wav_data = apply_gain(wav_data)

        return wav_data

    except Exception as e:
        print("Error processing request", e)
        return None

# Transcription function using faster-whisper
def transcribe(audio_file, lang):
    print("Transcribing")
    generator, _ = listen.transcribe(audio_file, language=lang, without_timestamps=True)
    transcription = " ".join(segment.text for segment in generator)
    print(transcription)
    return transcription

# Currency detection function
def ind_currency_detect(source):
    results = ind_currency_model(Image.open(io.BytesIO(source)), verbose=False)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in currency_map:
                return currency_map[class_id]
    return "No currency found!"

# Flask route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.headers.get('Content-Type') != 'application/octet-stream':
            return 'Invalid Content-Type', 400

        request_data = request.stream.read()
        if not request_data:
            return 'Empty request', 400

        response_data = process_request(request_data)
        if response_data:
            response = Response(response_data, mimetype='audio/wav')
            response.headers['Content-Length'] = str(len(response_data))
            return response
        else:
            return 'Error processing request', 500
    except Exception as e:
        print("Error handling request", e)
        return jsonify({"error": "Internal Server Error"}), 500

# Main entry point
if __name__ == '__main__':
    app.secret_key = secrets.token_urlsafe(32)
    app.run(host='172.20.234.190', port=8080, debug=False)