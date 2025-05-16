# Use this script to download and organize data from the hub for EmergentTTS-Evaluation

import os
from datasets import load_dataset
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import urllib.request
import json

dataset_name = "bosonai/EmergentTTS-Eval"
AUDIO_SAVE_PATH = "data/baseline_audios"
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

dataset_hf = load_dataset(dataset_name, split="train")
jsonl_data = []
for i in tqdm(range(len(dataset_hf))):
    curr_data = dataset_hf[i]
    jsonl_data.append({
        "category": curr_data["category"],
        "text_to_synthesize": curr_data["text_to_synthesize"],
        "evolution_depth": curr_data["evolution_depth"],
        "language": curr_data["language"],
    })
    # Create the pydub AudioSegment and save it to 
    audio_array = curr_data["audio"]["array"]
    audio_array_int16 = np.int16(audio_array * 32767)
    # Convert audio array to to 16-bit PCM
    audio_sample_rate = curr_data["audio"]["sampling_rate"]
    audio_segment = AudioSegment(
        audio_array_int16.tobytes(),
        frame_rate=audio_sample_rate,
        sample_width=2,
        channels=1
    )
    audio_segment.export(os.path.join(AUDIO_SAVE_PATH, f"{i}.wav"), format="wav")

# Save the jsonl data
with open("data/emergent_tts_eval_data.jsonl", "w") as f:
    for data in jsonl_data:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Jsonl data saved, now downloading the wv_mos.ckpt file from https://github.com/AndreevP/wvmos")
# Download the wv_mos.ckpt file
url = "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1"
path = "data/wv_mos.ckpt"
urllib.request.urlretrieve(
        url,
        path
    )
print("Done!!")