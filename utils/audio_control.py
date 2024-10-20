import os

import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from scipy.io import wavfile

from .file_control import get_base_name


def extract_audio(video_path, audio_dir: str = "audio") -> None:
    audio_path = os.path.join(audio_dir, get_base_name(video_path) + ".wav")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    video.close()
    audio.close()
    return None

def load_audio(audio_path: str) -> tuple:
    audio = AudioSegment.from_wav(audio_path)
    data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    return data, sample_rate