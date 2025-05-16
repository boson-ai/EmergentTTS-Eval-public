import os
import torch
import numpy as np
import io
from pydub import AudioSegment
import hashlib
import re
from whisper_normalizer.english import EnglishTextNormalizer
from wer_utils.evaluate_tokenizer import EvaluationTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import editdistance as ed
import librosa
english_normalizer = EnglishTextNormalizer()

# Constants
PUNCS = '!,.?;:'
TARGET_RATE = 16000
UNIQUE_KEY_NAME = "unique_id_eval"
GENERATION_CONFIG = {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 2048, "max_audio_new_tokens": 8192}

class AccelerateSyncBlock:
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def __enter__(self):
        self.accelerator.wait_for_everyone()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.wait_for_everyone()


def set_seed(seed=42) -> None:
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_wav_in_memory(audio_array):
    """
    To convert audio waveform array to a wav buffer in memory
    """
    assert audio_array.ndim == 1, "Audio array must be 1D"
    audio_int16 = (audio_array * 32767).astype(np.int16)
    audio_segment = AudioSegment(audio_int16.tobytes(), frame_rate=TARGET_RATE, sample_width=2, channels=1)
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()

def get_audio_array_from_path(audio_path):
    wv, _ = librosa.load(audio_path, sr=TARGET_RATE, mono=True)
    return wv

def calculate_file_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_dir_md5(dir_path):
    """Calculate the MD5 hash of all files in a directory."""
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(dir_path):
        for fname in sorted(files):  # Sort files to ensure consistent order
            file_path = os.path.join(root, fname)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_dedup_results(data):
    unique_ids = set()
    out = []
    for item in data:
        unique_id = item[UNIQUE_KEY_NAME]
        if unique_id not in unique_ids:
            unique_ids.add(unique_id)
            out.append(item)
    return out

def add_unique_id_to_samples(data):
    for i,item in enumerate(data):
        if UNIQUE_KEY_NAME in item:
            assert item[UNIQUE_KEY_NAME] == i, f"UNIQUE_KEY_NAME:{UNIQUE_KEY_NAME} already exists in the dataset and doesn't match the index"  
        item[UNIQUE_KEY_NAME] = i
    return data

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)  # 将文本中的连续空格替换为单个空格
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

def calculate_wer_qwen(gt: str, pred: str, language: str = "en") -> tuple[float, float]:
    # Adopted from https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_asr.py#L126-L162
    assert language in ["en", "zh"], "Language not supported"
    gt = gt.lower()
    pred = pred.lower()
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
            tokenizer_type="none",
            lowercase=True,
            punctuation_removal=True,
            character_tokenization=False,
        )
    gt = remove_sp(gt, language)
    pred = remove_sp(pred, language)
    gt = english_normalizer(gt)
    pred = english_normalizer(pred)
    gt_items = tokenizer.tokenize(gt).split()
    pred_items = tokenizer.tokenize(pred).split()
    distance = ed.eval(gt_items, pred_items)
    ref_length = len(gt_items)
    return distance, ref_length

def get_array_from_pydub_segment(audio_segment):
    # Convert audio to required format: 16kHz, mono
    audio_segment = audio_segment.set_frame_rate(TARGET_RATE).set_channels(1).set_sample_width(2)

    # Convert pydub audio to a NumPy array
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0  # Normalize PCM data

    return samples

class WhisperTranscriber:
    def __init__(self, model_path, device):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        self.model.eval()
        
    def transcribe(self, pydub_segmet):
        audio_array = get_array_from_pydub_segment(pydub_segmet)
        input_features = self.processor(audio_array, sampling_rate=TARGET_RATE, return_tensors="pt", return_attention_mask = True, truncation = False, padding="longest")
        input_features = input_features.to(self.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        predicted_ids = self.model.generate(**input_features, forced_decoder_ids=forced_decoder_ids, return_timestamps = True)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()