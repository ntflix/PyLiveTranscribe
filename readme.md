# PyLiveTranscribe

Demo of using [Whisper](https://github.com/openai/whisper) in Python to transcribe mic audio in real time, locally.

Bit flawed because it chops up the audio into x-second blocks, so anything said in between blocks will probably be wrong or not transcribed at all.

Run `main_continuous.py` for a continuous stream of audio, or `single_block.py` for a single 5-second block of audio.

Requires:
- whisper (and numpy)
- sounddevice
