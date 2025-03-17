import requests
import os
SERVER_URL = 'http://192.168.71.167:8080/upload'
image_file = 'test.jpg'
audio_file = 'audio2.wav'

# Read the image and audio files
with open(image_file, 'rb') as img_f, open(audio_file, 'rb') as audio_f:
    image_data = img_f.read()
    audio_data = audio_f.read()

user_id = b'user1234'
action = b'C'

# Concatenate the data with delimiters
data = image_data + b'|*|' + b'audio_data' + b'|*|' + user_id + b'|*|' + action

headers = {'Content-Type': 'application/octet-stream'}

try:
    response = requests.post(SERVER_URL, data=data, headers=headers, timeout=30)
    response.raise_for_status()

    # Extract the content from the response
    response_content = response.content

    # Save the received WAV data directly
    with open('123456.wav', 'wb') as response_file:
        response_file.write(response_content)

    print("Response received and saved successfully.")
    os.system('123456.wav')
except requests.exceptions.RequestException as e:
    print("Error:", e)
