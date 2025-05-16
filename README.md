# EmergentTTS-Eval
Code accompanying the paper, "EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-Judge"

# General Instructions
- The accompanying code provides the pipeline for reproducing the evaluation results for the models evaluated in the paper, and evaluating your own model on our dataset.
- To evaluate your own model, implement the `generate_audio_out` and `prepare_emergent_tts_sample` methods. If you want to load your model locally on GPU, implement it in `model_clients.py` file otherwise if it's an API call, put it in `api_clients.py`. **In either case, expose your client in the `evaluation_runner.py` file following the structure of other clients**.
- We use accelerate for local inference, so if you implement in `model_clients.py`, ensure to place the model on `accelerator.device`.
# Setup Instructions
First, clone the repo, create a virtual environment, and install the requirements.
```bash
git clone https://github.com/boson-ai/EmergentTTS-Eval
cd EmergentTTS-Eval
python3 -m venv .test_env
source .test_env/bin/activate
pip3 install -r requirements.txt
```
Then, download all of the relevant data using `download_data.py`, this processes the baseline audio files, creates a jsonl structure for the data, and downloads the wv_mos checkpoint for MOS calculation.
```bash
python3 download_data.py
```
> [!NOTE]
> If you are evaluating your own model or a new model from another provider, install the specific requirements for that model along with already installed requirements.txct, as the current requirements only cover the setup for the benchmark.

# Evaluation Instructions
## Local Inference with Accelerate
If the model is in `model_clients.py`, evaluation will be run with accelerate. For example, to reproduce Orpheus-TTS results, use the command:
```bash
MODEL_NAME="orpheus-tts-0.1-finetune-prod"
export HF_TOKEN=<your_hf_token>
export JUDGER_API_KEY=<api key for the judger model, either gemini or openai model>
accelerate launch --config_file default_accelerate_config.yaml evaluation_runner.py \
--model_name_or_path "canopylabs/orpheus-tts-0.1-finetune-prod" \
--output_dir "<path_to_output_dir, absolute path is recommended>" \
--seed 42 \
--judge_model_provider "gemini-2.5-pro-preview-05-06" \
--temperature 1.0 \
--tts_judger_evaluate_function "win_rate" \
--baseline_audios_path <path/to/EmergentTTS-Eval/data/baseline_audios, these will be stored when you run download_data.py>
```

## Inference for API-served models
For models in `api_clients.py`, we call the api with multiple threads for faster inference, the MOS model and WhisperV3 for transcription still needs to be placed in the GPU for this.
```bash
export <>_API_KEY=<key for model to evaluate, for openai models it's OPENAI_API_KEY, for deepgram, it's DEEPGRAM_API_KEY, for custom added api_client, follow the variable name you implement in api_clients.py> 
export JUDGER_API_KEY=<api key for the judger model, either gemini or openai model>
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u evaluation_runner.py \
--model_name_or_path "gpt-4o-mini-tts" \
--output_dir "<path_to_output_dir, absolute path is recommended>" \
--seed 42 \
--judge_model_provider "gemini-2.5-pro-preview-05-06" \
--api_num_threads 14 \
--temperature 1.0 \
--text_speech_strong_prompting
--tts_judger_evaluate_function "win_rate" \
--baseline_audios_path <path/to/EmergentTTS-Eval/data/baseline_audios, these will be stored when you run download_data.py>
```

# Flags and Considerations
- For single GPU setup, edit the `default_accelerate_config.yaml` file and use `gpu_ids: 0,` and `num_processes: 1`.
- Use the `--num_samples` parameter to evaluate on only a subset of samples.
- Carefully tweak `--api_num_threads` and `CUDA_VISIBLE_DEVICES` for api client based inference in your setup. We place `--api_num_threads` instances of the MOS model and WhisperV3 uniformly in all visible gpus, so too high `--api_num_threads` can result in Cuda OOM.
- Omit passing the `--tts_judger_evaluate_function` if you do not want to use LALM judger to judge the output, and only calculate other metrics like WER, MOS, etc.
- Pass `--text_speech_strong_prompting` if the client supports strong_prompting, as described in the paper.
- To evaluate only on specific depths or categories, use the `--depths_to_evaluate` and `--categories_to_evaluate` flags.
- To use a specific voice of the model, if applicable, use the `--voice_to_use` paramter.
- `--baseline_audios_path` needs to be passed if win_rate is calculated.
- To disable audio generation and fetch generated audios from a path, use the `--fetch_audios_from_path` paramter, this can be useful for testing different judger and not changing the generated audios.
