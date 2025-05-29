# EmergentTTS-Eval
Code accompanying the paper, "EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge"

# Introduction
Text-to-Speech (TTS) benchmarks often fail to capture how well models handle nuanced and semantically complex text. We introduce *EmergentTTS-Eval*, a comprehensive benchmark covering six challenging TTS scenarios: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation (e.g. URLs, formulas), and questions. Crucially, our framework automates both test-case generation and evaluation, making the benchmark easily extensible. Starting from a small set of human-written seed prompts, we iteratively extend them using LLMs to target specific structural, phonetic and prosodic challenges, resulting in 1,645 diverse test cases. Moreover, we employ a model-as-a-judge approach, using a Large Audio Language Model (LALM) to assess the speech across multiple dimensions such as expressed emotion, prosodic, intonational, and pronunciation accuracy. We evaluate state-of-the-art open-source and proprietary TTS systems, such as 11Labs, Deepgram, and OpenAI's 4o-mini-TTS, on EmergentTTS-Eval, demonstrating its ability to reveal fine-grained performance differences. Results show that the model-as-a-judge approach offers robust TTS assessment and a high correlation with human preferences.

# Leaderboard
### WER ↓ and Win-rate ↑ over all categories with gpt-4o-mini-tts-alloy as baseline and gemini-2.5-pro as judger; † indicates Strong Prompted models.
> [!NOTE]
> The generated audios for all of the models we evaluated, along with the predictions made by the gemini-2.5-pro judger are available in this [Google Drive link](https://drive.google.com/drive/folders/1SGEGaUai2UqOMbwXx447yZeY-6gCU0F_?usp=drive_link).

| Model                        | Voice            | Overall WER ↓ | Overall Win-Rate ↑ | Emotions WER ↓ | Emotions Win-Rate ↑ | Foreign Words WER ↓ | Foreign Words Win-Rate ↑ | Paralinguistics WER ↓ | Paralinguistics Win-Rate ↑ | Complex Pronunciation WER ↓ | Complex Pronunciation Win-Rate ↑ | Questions WER ↓ | Questions Win-Rate ↑ | Syntactic Complexity WER ↓ | Syntactic Complexity Win-Rate ↑ |
|------------------------------|------------------|----------------|---------------------|---------------------|--------------------------|-----------------------|----------------------------|-----------------------------|----------------------------------|-----------------|----------------------|-----------------------------|---------------------------------|---------------|--------------------|
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)        | Ballad           | 11.87         | 65.17%            | 1.82           | 88.84%              | 13.30               | 60.17%                   | 21.15                 | 82.14%                     | 35.32                       | 40.40%                           | 1.38            | 56.96%               | 1.16                        | 59.53%                         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)        | Alloy            | 12.00         | 57.95%            | 0.93           | 61.64%              | 13.75               | 62.50%                   | 20.56                 | 68.21%                     | 36.92                       | 49.59%                           | 1.72            | 47.85%               | 1.26                        | 56.85%                         |
| [gpt-4o-mini-tts†](https://platform.openai.com/docs/models/gpt-4o-mini-tts)             | Alloy            | 10.76         | 56.32%            | 0.71           | 59.17%              | 12.07               | 57.32%                   | 21.33                 | 58.75%                     | 31.57                       | 52.44%                           | 0.66            | 52.67%               | 0.84                        | 57.14%                         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)         | Alloy            | 12.38         | 53.76%            | 1.03           | 48.57%              | 14.72               | 60.17%                   | 23.16                 | 66.78%                     | 35.89                       | 40.81%                           | 1.19            | 47.50%               | 1.25                        | 57.14%                         |
| [gpt-4o-mini-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview)   | Alloy            | 13.09         | 52.31%            | 9.34           | 59.13%              | 12.70               | 58.92%                   | 20.92                 | 62.59%                     | 37.14                       | 28.68%                           | 0.74            | 48.21%               | 0.72                        | 53.40%                         |
| [gpt-4o-mini-tts (baseline)](https://platform.openai.com/docs/models/gpt-4o-mini-tts)   | Alloy            | 10.61         | –                 | 0.72           | –                   | 13.45               | –                        | 20.55                 | –                          | 29.90                       | –                                | 0.42            | –                    | 1.04                        | –                               |
| [gpt-4o-mini-audio-preview](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview)    | Alloy            | 10.92         | 49.60%            | 0.95           | 55.89%              | 14.48               | 59.82%                   | 19.04                 | 52.86%                     | 32.27                       | 30.61%                           | 0.55           | 47.32%               | 0.88                        | 48.75%                         |
| [HumeAI†](https://www.hume.ai/research)                      | –                | 12.85         | 42.73%            | 0.83           | 61.60%              | 21.05               | 34.64%                   | 19.84                 | 36.91%                     | 37.14                       | 34.28%                           | 0.38            | 43.21%               | 0.93                        | 44.64%                         |
| [11Labs eleven multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)| Brian            | 11.19         | 33.89%            | 0.63           | 30.35%              | 14.44               | 35.53%                   | 21.51                 | 45.53%                     | 31.44                       | 14.48%                           | 0.49            | 39.46%               | 1.15                        | 35.53%                         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)              | Thalia-en        | 16.83         | 32.44%            | 3.45           | 29.28%              | 21.41               | 18.75%                   | 23.73                 | 21.14%                     | 54.49                       | 33.81%                           | 1.24            | 48.21%               | 1.36                        | 43.70%                         |
| [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS?tab=readme-ov-file)                  | Tara             | 17.71         | 30.12%            | 1.81           | 31.78%              | 22.31               | 17.50%                   | 40.94                 | 39.82%                     | 41.04                       | 10.61%                           | 1.48            | 39.64%               | 1.63                        | 38.92%                         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)               | Chelsie          | 23.03         | 28.77%            | 2.41           | 41.60%              | 26.77               | 11.42%                   | 58.44                 | 20.25%                     | 49.51                       | 6.12%                            | 0.87            | 51.78%               | 3.47                        | 38.57%                         |
| [Qwen 2.5 Omni](https://github.com/QwenLM/Qwen2.5-Omni)                | Chelsie          | 26.58         | 27.07%            | 1.22           | 41.18%              | 26.98               | 11.07%                   | 57.48                 | 17.44%                     | 64.07                       | 3.30%                            | 12.77           | 49.28%               | 1.66                        | 36.96%                         |
| [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)                      | –                | 31.40         | 22.36%            | 12.36          | 31.83%              | 33.46               | 6.42%                    | 58.48                 | 21.50%                     | 82.15                       | 1.84%                            | 5.21            | 32.50%               | 3.08                        | 37.50%                         |
| [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts)                 | random           | 28.62         | 17.67%            | 13.04          | 17.92%              | 29.61               | 10.00%                   | 64.93                 | 14.28%                     | 51.87                       | 1.59%                            | 10.44           | 28.28%               | 6.35                        | 30.82%                         |
| [Zyphra/Zonos](https://www.zyphra.com/post/beta-release-of-zonos-v0-1)                 | exampleaudio     | 19.12         | 16.55%            | 7.32           | 9.67%               | 28.52               | 11.96%                   | 25.33                 | 13.75%                     | 45.00                       | 7.95%                            | 7.66            | 26.78%               | 4.13                        | 28.13%                         |
| [Sesame1B](https://github.com/SesameAILabs/csm)                     | –                | 32.32         | 15.96%            | 17.07          | 7.32%               | 45.27               | 10.35%                   | 49.63                 | 18.92%                     | 80.97                       | 7.40%                            | 2.74            | 31.78%               | 4.30                        | 18.88%                         |
| [Suno Bark](https://github.com/suno-ai/bark)                    | v2/en_speaker_6  | 20.71         | 8.90%             | 4.31           | 0.00%               | 26.11               | 10.89%                   | 33.26                 | 6.60%                      | 55.88                       | 8.36%                            | 3.01            | 15.00%               | 6.07                        | 12.50%                         |

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
> If you are evaluating your own model or a new model from another provider, install any missing requirements for that model along with already installed requirements.txt, as the current requirements only cover the ones necessary for the benchmark and judger sdk's - google-generativeai and openai.

# Evaluation Instructions

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
--text_speech_strong_prompting \
--tts_judger_evaluate_function "win_rate" \
--baseline_audios_path <path/to/EmergentTTS-Eval/data/baseline_audios, these will be stored when you run download_data.py>
```


## Local Inference with Accelerate
If the model is in `model_clients.py`, evaluation will be run with accelerate. For example, to reproduce Orpheus-TTS results, install it's specific requirements(`snac` is the only missing requirement for orpheus) and use the command:
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

# Custom Judger Model
To use your own Judger Model, we recommend using vLLM to create an OpenAI compatible server. Next, you just have to pass the flag `--judger_base_url` pointing to the IP and port of the custom server.

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
