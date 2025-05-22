import whisper
import sys

if len(sys.argv) != 2:
    print("Usage: python transcribe_audio.py path/to/audio.wav")
    sys.exit(1)

audio_path = sys.argv[1]
model = whisper.load_model("base")  # or use "tiny" for faster

result = model.transcribe(audio_path)
print("\n🔊 Transcription:")
print(result["text"])
