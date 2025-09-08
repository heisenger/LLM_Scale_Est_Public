import base64
import subprocess
from pathlib import Path


def mp3_to_base64_wav(mp3_path):
    """
    Converts MP3 to WAV (16kHz mono) in memory and returns base64 string.
    Requires ffmpeg installed.
    """
    wav_bytes = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(mp3_path),
            "-ac",
            "1",
            "-ar",
            "16000",  # mono, 16kHz
            "-f",
            "wav",
            "-",  # output WAV to stdout
        ]
    )
    return base64.b64encode(wav_bytes).decode("utf-8")


def mp3_to_base64(mp3_path: str) -> str:
    """Read MP3 file and return base64-encoded string."""
    mp3_bytes = Path(mp3_path).read_bytes()
    return base64.b64encode(mp3_bytes).decode("utf-8")


import io
import soundfile as sf  # pip install soundfile


def base64_wav_to_array(base64_str):
    """
    Decodes a base64-encoded WAV string into (numpy_array, sample_rate).
    """
    wav_bytes = base64.b64decode(base64_str)
    with io.BytesIO(wav_bytes) as buf:
        audio_array, sr = sf.read(buf, dtype="float32")
    return audio_array, sr


import base64
from pathlib import Path


def base64_to_mp3(base64_str: str, output_path: str):
    """Decode a base64 string and save as MP3 file."""
    mp3_bytes = base64.b64decode(base64_str)
    Path(output_path).write_bytes(mp3_bytes)
