import sounddevice as sd  # type: ignore
import numpy as np
from typing import Any
from typing import AsyncGenerator
import asyncio

from transcribe import Transcribe


DURATION_BLOCKS: int = 10  # seconds
SAMPLE_RATE: int = 16000
transcribe = Transcribe()


def streamCallback(
    indata: Any,
    frames: int,
    time: Any,
    status: Any,
) -> None:
    result = transcribe.transcribe(
        indata,
        DURATION_BLOCKS,
        SAMPLE_RATE,
    )
    if result.no_speech_prob < 0.5:
        print(result.text, end=" ")
    else:
        pass
        # print("No speech detected")


async def transcribeContinuous(
    channels: int = 1, **kwargs: Any
) -> AsyncGenerator[
    tuple[
        np.ndarray[
            np.float64,
            np.dtype[np.float64],
        ],
        sd.CallbackFlags,
    ],
    None,
]:
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in: asyncio.Queue[
        tuple[
            np.ndarray[
                np.float64,
                np.dtype[np.float64],
            ],
            sd.CallbackFlags,
        ]
    ] = asyncio.Queue()
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

    def callback(
        indata: np.ndarray[
            np.float64,
            np.dtype[np.float64],
        ],
        frame_count: int,
        time_info: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream: sd.InputStream = sd.InputStream(
        callback=callback,
        channels=channels,
        samplerate=SAMPLE_RATE,
        blocksize=int(DURATION_BLOCKS * SAMPLE_RATE),
        **kwargs,
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def main() -> None:
    async for indata, status in transcribeContinuous():
        streamCallback(
            indata=indata,
            frames=indata.shape[0],
            time=sd.CallbackFlags(),
            status=status,
        )


if __name__ == "__main__":
    asyncio.run(main())
