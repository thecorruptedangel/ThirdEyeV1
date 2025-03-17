import os
from piper import Piper

# Path to the model directory
model_path = os.path.expanduser("~/piper_models/en_us")

# Initialize Piper TTS
piper = Piper(model_path)

# Text to convert to speech
text = "Hello, this is a test of Piper TTS on a Raspberry Pi."

# Generate speech
audio = piper.tts(text)

# Save the output to a file
with open("output.wav", "wb") as f:
    f.write(audio)

print("Speech synthesis complete. Check the output.wav file.")
