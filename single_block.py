from transcribe import Transcribe
import sounddevice as sd  # type: ignore
from typing import Any

a = Transcribe()
duration = 5  # seconds
sample_rate = 16000


def streamCallback(
    indata: Any,
    outdata: Any,
    frames: int,
    time: Any,
    status: Any,
) -> None:
    result = a.transcribe(
        indata,
        duration,
        sample_rate,
    )
    print(result.text)


# constantly record, pass to transcribe, print result
while True:
    with sd.InputStream(  # type: ignore
        channels=1,
        callback=streamCallback,
        samplerate=sample_rate,
        blocksize=int(duration * sample_rate),
    ):
        sd.sleep(int(1))  # type: ignore
