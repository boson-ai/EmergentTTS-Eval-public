<h1 align="center">EmergentTTS-Eval</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2505.23009"><img src="https://img.shields.io/badge/arXiv-2505.23009-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/bosonai/EmergentTTS-Eval"><img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg" alt="Hugging Face"></a>
</p>

Code accompanying the paper, "EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge"

# Introduction
Text-to-Speech (TTS) benchmarks often fail to capture how well models handle nuanced and semantically complex text. We introduce *EmergentTTS-Eval*, a comprehensive benchmark covering six challenging TTS scenarios: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation (e.g. URLs, formulas), and questions. Crucially, our framework automates both test-case generation and evaluation, making the benchmark easily extensible. Starting from a small set of human-written seed prompts, we iteratively extend them using LLMs to target specific structural, phonetic and prosodic challenges, resulting in 1,645 diverse test cases. Moreover, we employ a model-as-a-judge approach, using a Large Audio Language Model (LALM) to assess the speech across multiple dimensions such as expressed emotion, prosodic, intonational, and pronunciation accuracy. We evaluate state-of-the-art open-source and proprietary TTS systems, such as 11Labs, Deepgram, and OpenAI's 4o-mini-TTS, on EmergentTTS-Eval, demonstrating its ability to reveal fine-grained performance differences. Results show that the model-as-a-judge approach offers robust TTS assessment and a high correlation with human preferences.

# Updates
- [8-10-25] EmergentTTS-Eval has been accepted for publication in the datasets and benchmark track at NeurIPS'25 🎉.
- [8-10-25] We have updated our leaderboard with the latest gemini-2.5-pro model, check note under the leaderboard section for more details.

# Leaderboard
### WER ↓ and Win-rate ↑ over all categories with gpt-4o-mini-tts-alloy as baseline and gemini-2.5-pro as judger; ⭐ indicates that the model was strongly prompted(see paper for definition). For some expensive to use models, we only do strong prompting.
> [!NOTE]
> The generated audios for all of the models we evaluated, along with the predictions made by the gemini-2.5-pro judger are available in this [Google Drive link](https://drive.google.com/drive/folders/1SGEGaUai2UqOMbwXx447yZeY-6gCU0F_?usp=drive_link).

> [!NOTE]
> The original leaderboard was created using gemini-2.5-pro-preview-05-06 as judge, since the model has been depreciated, we have moved the leaderboard to LEADERBOARD_gemini-2.5-pro-05-06.md. The current leaderboard is with the non-preview model pointed at gemini-2.5-pro, thinking budget=256. Both leaderboards have a spearmann correlation of 0.98 at p-value <0.05 indicating strong positive ranking correlation.

| Model                        | Voice            | Overall WER ↓ | Overall Win-Rate ↑ | Emotions WER ↓ | Emotions Win-Rate ↑ | Foreign Words WER ↓ | Foreign Words Win-Rate ↑ | Paralinguistics WER ↓ | Paralinguistics Win-Rate ↑ | Complex Pronunciation WER ↓ | Complex Pronunciation Win-Rate ↑ | Questions WER ↓ | Questions Win-Rate ↑ | Syntactic Complexity WER ↓ | Syntactic Complexity Win-Rate ↑ |
|------------------------------|------------------|----------------|---------------------|---------------------|--------------------------|-----------------------|----------------------------|-----------------------------|----------------------------------|-----------------|----------------------|-----------------------------|---------------------------------|---------------|--------------------|
| [Gemini-2.5-Flash-Preview-TTS](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview-tts)⭐       | Zephyr           | 10.39         | 75.57%            | 0.71           | 97.32%              | 11.80               | 65.56%                   | 18.38                | 91.25%                     | 33.57                       | 55.73%                           | 0.40            | 74.82%               | 0.50                        | 67.14%                         |
| [Gemini-2.5-Flash-Preview-TTS](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview-tts)        | Zephyr           | 10.32         | 75.39%            | 0.60           | 95.00%              | 12.99               | 63.03%                   | 18.25                | 89.10%                     | 31.90                       | 60.33%                           | 0.36            | 74.82%               | 0.73                        | 68.03%                         |
| [Gemini-2.5-Pro-Preview-TTS](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-tts)⭐        | Zephyr           | 11.79         | 72.15%            | 0.87           | 92.14%              | 16.22               | 63.44%                   | 20.87                | 84.28%                     | 34.75                       | 66.45%                           | 0.72            | 65.71%               | 0.87                        | 60.00%                         |
| [gpt-4o-audio-preview](https://platform.openai.com/docs/models/gpt-4o-audio-preview)⭐        | Ballad           | 11.87         | 72.67%            | 1.82           | 96.96%              | 13.30               | 70.00%                   | 21.15                 | 89.46%                     | 35.32                       | 45.30%                           | 1.38            | 62.50%               | 1.16                        | 68.39%                         |
| [gpt-4o-audio-preview](https://platform.openai.com/docs/models/gpt-4o-audio-preview)⭐        | Alloy            | 12.00         | 62.97%            | 0.93           | 67.14%              | 13.75               | 71.60%                   | 20.56                 | 77.32%                     | 36.92                       | 51.83%                           | 1.72            | 49.64%               | 1.26                        | 58.92%                         |
| [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)⭐             | Alloy            | 10.76         | 60.24%            | 0.71           | 67.32%              | 12.07               | 63.03%                   | 21.33                 | 63.39%                     | 31.57                       | 50.20%                           | 0.66            | 54.28%               | 0.84                        | 61.96%                         |
| [gpt-4o-audio-preview](https://platform.openai.com/docs/models/gpt-4o-audio-preview)         | Alloy            | 12.38         | 58.14%            | 1.03           | 56.78%              | 14.72               | 68.75%                   | 23.16                 | 71.60%                     | 35.89                       | 44.48%                           | 1.19            | 48.75%               | 1.25                        | 56.78%                         |
| [gpt-4o-mini-audio-preview](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview)⭐   | Alloy            | 13.09         | 61.03%            | 9.34           | 74.10%              | 12.70               | 73.39%                   | 20.92                 | 68.03%                     | 37.14                       | 31.02%                           | 0.74            | 55.53%               | 0.72                        | 60.35%                         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts) | Alloy | 10.61 | 50% | 0.72 | – | 13.45 | – | 20.55 | – | 29.90 | – | 0.42 | – | 1.04 | – |
| [gpt-4o-mini-audio-preview](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview)    | Alloy            | 10.92         | 52.18%            | 0.95           | 62.90%              | 14.48               | 64.82%                   | 19.04                 | 52.14%                     | 32.27                       | 35.30%                           | 0.55           | 45.71%               | 0.88                        | 50.17%                         |
| [HumeAI](https://www.hume.ai/research)⭐                      | –                | 12.85         | 46.77%            | 0.83           | 71.78%              | 21.05               | 41.25%                   | 19.84                 | 42.14%                     | 37.14                       | 33.06%                           | 0.38            | 42.14%               | 0.93                        | 48.57%                         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)                      | English_expressive_narrator                | 10.07         | 36.77%            | 0.57           | 41.60%              | 14.58               | 31.96%                   | 17.69                 | 33.57%                     | 28.77                       | 12.85%                           | 0.27            | 52.14%               | 0.84                        | 45.53%                         |
| [11Labs eleven multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)| Brian            | 11.19         | 33.64%            | 0.63           | 35.35%              | 14.44               | 36.60%                   | 21.51                 | 52.14%                     | 31.44                       | 14.69%                           | 0.49            | 28.21%               | 1.15                        | 32.50%                         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)              | Thalia-en        | 16.83         | 25.80%            | 3.45           | 17.50%              | 21.41               | 15.89%                   | 23.73                 | 20.89%                     | 54.49                       | 29.18%                           | 1.24            | 43.03%               | 1.36                        | 28.75%                         |
| [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS?tab=readme-ov-file)                  | Tara             | 17.71         | 29.44%            | 1.81           | 39.06%              | 22.31               | 14.64%                   | 40.94                 | 48.57%                     | 41.04                       | 11.63%                           | 1.48            | 31.07%               | 1.63                        | 29.46%                         |
| [Qwen 2.5 Omni](https://github.com/QwenLM/Qwen2.5-Omni)⭐               | Chelsie          | 23.03         | 28.99%            | 2.41           | 46.60%              | 26.77               | 14.46%                   | 58.44                 | 21.25%                     | 49.51                       | 6.12%                            | 0.87            | 48.92%               | 3.47                        | 33.75%                         |
| [Qwen 2.5 Omni](https://github.com/QwenLM/Qwen2.5-Omni)                | Chelsie          | 26.58         | 25.98%            | 1.22           | 41.07%              | 26.98               | 12.50%                   | 57.48                 | 20.89%                     | 64.07                       | 1.63%                            | 12.77           | 49.10%               | 1.66                        | 27.67%                         |
| [ResembleAI Chatterbox](https://github.com/resemble-ai/chatterbox)                | -          | 13.05         | 24.49%            | 1.18           | 28.03%              | 17.59               | 24.64%                   | 20.64                 | 17.32%                     | 40.75                       | 7.95%                            | 0.65           | 51.07%               | 0.96                        | 15.89%                         |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)                | af_heart          | 13.41         | 25.89%            | 0.71           | 18.03%              | 22.17               | 13.21%                   | 28.37                 | 5.89%                     | 30.10                       | 28.36%                            | 0.56           | 43.39%               | 0.65                        | 46.78%                         |
| [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)                      | –                | 31.40         | 17.02%            | 12.36          | 22.85%              | 33.46               | 5.89%                    | 58.48                 | 21.60%                     | 82.15                       | 1.42%                            | 5.21            | 22.14%               | 3.08                        | 26.25%                         |
| [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts)                 | random           | 28.62         | 16.36%            | 13.04          | 20.96%              | 29.61               | 11.07%                   | 64.93                 | 14.64%                     | 51.87                       | 0.90%                            | 10.44           | 24.08%               | 6.35                        | 23.39%                         |
| [KyutAI-TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr)                 | ex03-ex01_happy_001_channel1_334s           | 12.72         | 21.94%            | 0.83         | 31.78%              | 16.47               | 13.39%                   | 26.41                | 24.28%                     | 33.43                       | 8.57%                            | 0.61           | 35.17%               | 1.19                        | 16.78%                         |
| [Zyphra/Zonos](https://www.zyphra.com/post/beta-release-of-zonos-v0-1)                 | exampleaudio     | 19.12         | 14.07%            | 7.32           | 6.96%               | 28.52               | 14.28%                   | 25.33                 | 14.82%                     | 45.00                       |4.08%                            | 7.66            | 25.71%               | 4.13                        | 17.32%                         |
| [Sesame1B](https://github.com/SesameAILabs/csm)                     | –                | 32.32         | 13.37%            | 17.07          | 11.42%               | 45.27               | 5.00%                   | 49.63                 | 20.53%                     | 80.97                       | 5.91%                            | 2.74            | 21.25%               | 4.30                        | 15.17%                         |
| [F5TTS](https://github.com/SWivid/F5-TTS)                     | basic_ref_en                |   16.47       | 17.11%            | 0.70          | 29.64%               | 23.51               | 3.21%                   | 31.66                 | 19.82%                     | 42.41                       | 3.87%                            | 1.62            | 19.46%               | 2.14                        | 25.00%                         |
| [Suno Bark](https://github.com/suno-ai/bark)                    | v2/en_speaker_6  | 20.71         | 9.02%             | 4.31           | 2.14%               | 26.11               | 8.57%                   | 33.26                 | 12.50%                      | 55.88                       | 7.14%                            | 3.01            | 10.53%               | 6.07                        | 13.03%                         |
| [VITS-VCTK](https://github.com/jaywalnut310/vits)                    | default  | 27.45         | 8.47%             | 16.34           | 1.78%               | 47.45               | 6.72%                   | 51.12                 | 3.76%                      | 44.30                       | 16.53%                            | 2.37            | 17.14%               | 5.24                        | 5.79%                         |

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

If our work proves useful to you, cite it as:
```bash
@misc{manku2025emergentttseval,
    title={EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge},
    author={Ruskin Raj Manku and Yuzhi Tang and Xingjian Shi and Mu Li and Alex Smola},
    year={2025},
    eprint={2505.23009},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
