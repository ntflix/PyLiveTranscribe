import whisper  # type: ignore
import numpy as np


class Transcribe:
    __model: whisper.Whisper

    def __init__(self) -> None:
        self.__model = whisper.load_model("base")

    def transcribe(
        self,
        audio: np.ndarray[
            np.float64,
            np.dtype[np.float64],
        ],
        duration: int,
        sample_rate: int,
    ) -> whisper.DecodingResult:
        # return self.__model.transcribe(audio)
        audio = np.reshape(audio, (duration * sample_rate,))  # type: ignore

        padded_audio: np.ndarray[
            np.float64,
            np.dtype[np.float64],
        ] = whisper.pad_or_trim(  # type: ignore
            array=audio,
        )

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(padded_audio)  # type: ignore
        mel = mel.to(self.__model.device)

        # detect the spoken language
        # _, probs = self.__model.detect_language(mel)  # type: ignore
        # print(f"Detected language: {max(probs, key=probs.get)}")  # type: ignore

        # decode the audio
        options = whisper.DecodingOptions(
            fp16=False,
            language="en",
            suppress_blank=False,
        )
        # only does <=30s at a time:
        result = whisper.decode(self.__model, mel, options)
        # print(f"Decoded text: {result.text}")  # type: ignore
        return result  # type: ignore
