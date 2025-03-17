import secrets
import time
import io
import wave
import subprocess
import zlib
from flask import Flask, request, Response, jsonify, g
from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer, logging
from PIL import Image
from ultralytics import YOLO
from faster_whisper import WhisperModel
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment

lang_set = "ml-gtts"
# logging.basicConfig()
# logging.getLogger("faster_whisper").setLevel(logging.ERROR)
device, dtype = detect_device()
logging.set_verbosity_error()

listener_id = "distil-large-v3"
vision_model_id = "vikhyatk/moondream2"
tts_voice_id = "en_US-ljspeech-medium.onnx"
# tts_voice_id = "en_US-libritts_r-medium.onnx"
# tts_voice_id = "en_US-amy-medium.onnx"
tokenizer = AutoTokenizer.from_pretrained(vision_model_id, revision=LATEST_REVISION)
ind_currency_model = YOLO("ind_currency.pt")
vision = Moondream.from_pretrained(
    vision_model_id,
    revision=LATEST_REVISION,
    torch_dtype=dtype, attn_implementation="flash_attention_2"
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

def text2wav(phrase, model_name):
    command = f'echo """{phrase}""" | piper --model {model_name} --output-raw'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        return None
    audio_data = io.BytesIO(output)
    wav_data = audio_data.getvalue()
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Assuming mono audio
        wav_file.setsampwidth(2)   # Assuming 16-bit audio
        wav_file.setframerate(22050)  # Example frame rate
        wav_file.writeframes(wav_data)
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

def transcribe(audio_file, lang):
    print("transcribing")
    generator, _ = listen.transcribe(audio_file, language=lang, without_timestamps=True)
    transcription = " ".join(segment.text for segment in generator)
    print(transcription)
    return transcription

def ind_currency_detect(source):
    results = ind_currency_model(Image.open(io.BytesIO(source)), verbose=False)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if (class_id in currency_map):
                return currency_map[class_id]
    return "No currency found!"

def GetMalayalam(text):
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

def process_request(data):
    start_time = time.time()
    try:
        image_data, audio_data, user_id, action = data.split(b'|*|')

        print(f"UserID: {user_id.decode()}")
        print(f"Action: {action.decode()}")

        if action.decode() in ['A', 'B']:
            prompt = "Describe this image." if action.decode() == 'A' else transcribe(io.BytesIO(audio_data), "en")
            print("Prompt: " + prompt)
            result = vision.answer_question(vision.encode_image(Image.open(io.BytesIO(image_data))), prompt, tokenizer)

        elif action.decode() == 'C':
            prompt = "Currency Detection"
            print("Prompt: " + prompt)
            result = ind_currency_detect(image_data)
        else:
            print("Invalid Action")
            return None

        if(lang_set == "en-gtts"):
            wav_data = text2wav(result, tts_voice_id)
        elif(lang_set == "ml-gtts"):
            wav_data = GetMalayalam(result)
        else:
            wav_data = text2wav(result, tts_voice_id)

        print(f"Execution_Time: {time.time() - start_time} seconds")
        print("Result: "+result)
        
        return wav_data
        
    except Exception as e:
        print("Error processing request", e)
        return None


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
            return Response(response_data, mimetype='audio/wav')
        else:
            return 'Error processing request', 500
    except Exception as e:
        print("Error handling request", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.secret_key = secrets.token_urlsafe(32)
    app.run(host='0.0.0.0', port=8080, debug=False)
