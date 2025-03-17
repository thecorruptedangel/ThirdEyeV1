from googletrans import Translator
from gtts import gTTS
import io
import wave
from pydub import AudioSegment

def translate_and_convert_to_audio(text):
    # Translate English text to Malayalam
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='ml').text
    
    # Convert translated text to speech in MP3 format
    tts = gTTS(translated_text, lang='ml')

    # Save the speech as an MP3 file in memory
    audio_data_mp3 = io.BytesIO()
    tts.write_to_fp(audio_data_mp3)

    # Reset the pointer of the buffer to the beginning
    audio_data_mp3.seek(0)
    
    # Convert MP3 audio data to WAV format
    audio_segment = AudioSegment.from_mp3(audio_data_mp3)
    wav_data = audio_segment.raw_data
    
    return wav_data

def main():
    # Get text input from the user
    text = input("Enter the English text to translate and save as audio in Malayalam: ")

    # Translate and convert text to audio
    wav_data = translate_and_convert_to_audio(text)

    # Write the audio data to a WAV file
    with wave.open('test.wav', 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(22500)  # 22.5kHz
        wf.writeframes(wav_data)

    print("Audio saved as test.wav")

if __name__ == "__main__":
    main()
