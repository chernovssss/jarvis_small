import pyaudio
import sounddevice as sd


def get_audio_stream():
    return pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        output=True,
        frames_per_buffer=1024,
    )


def play_sound(sound):
    sd.play(sound, samplerate=16000)
    sd.wait()
