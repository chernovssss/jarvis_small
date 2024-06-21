import os
import sys

import numpy as np

from conversation import Chat
from online_wisper import FasterWhisperASR
from audio_stream import get_audio_stream, play_sound

from TTS.api import TTS
import sounddevice as sd
from supresser import stdchannel_redirected
from globals import *

stt = FasterWhisperASR(lan="ru", modelsize="large-v3")

stream = get_audio_stream()

tts = TTS(TTS_MODEl_PATH).to("cuda")

chat = Chat(LLAMA_MODEL_PATH, "Пользователь", "Идиот")


# stt.use_vad()

while True:
    record = []
    print("> Recording...")
    for _ in range(50):
        inp = stream.read(CHUNK_SIZE)
        inp = np.frombuffer(inp, dtype=np.int16)
        record.append(inp)

    print("> Processing...")
    inp = np.hstack(record)
    print("> Transcribing...")
    inp = stt.transcribe(inp)[0].text
    print("> Answering...")
    output = chat.prepare_message(inp)
    print(f"> {inp}")
    to_tts = []
    for x in output:
        print(x, end="")
        to_tts.append(x)
    to_tts = "".join(to_tts)
    print("> Speaking...")
    with stdchannel_redirected(sys.stderr, os.devnull):
        tts_output = tts.tts(
            text=to_tts,
            speaker_wav="test_audio.wav",
            language="ru",
            split_sentences=False,
        )
    play_sound(tts_output)
