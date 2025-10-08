import json
import os
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
from utils_eval import (
    set_seed,
    AccelerateSyncBlock,
    calculate_file_md5,
    get_dedup_results,
    add_unique_id_to_samples,
    UNIQUE_KEY_NAME,
    GENERATION_CONFIG,
    get_audio_array_from_path,
    WhisperTranscriber,
    calculate_wer_qwen,
)
from prompts import (
    EmergentTextSpeech
)
from quality_assessment.wv_mos import Wav2Vec2MOS
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json_repair
import traceback
from collections import defaultdict
from threading import Lock
from pydub import AudioSegment
from api_clients import GeminiClient, OpenAIClient

DATA_BASE_PATH = "data/"
md5_hash_value = "3fe115352376ebae9107dd79d3781edd"

source_language_to_hf_transcriber = {
    "en": (WhisperTranscriber,"openai/whisper-large-v3")
}

@dataclass
class EmergentTTSEvalConfig:
    """Configuration for Task evaluation"""
    prompting_object: EmergentTextSpeech
    eval_data_path: str
    metrics_path: str = ""
    predictions_path: str = ""
    output_audios_dir: str = ""

    @classmethod
    def create(cls, data_dir_path: str, output_dir_path: Optional[str] = None, strong_prompting: bool = False, voice_to_use: Optional[str] = None) -> "EmergentTTSEvalConfig":
        config = cls(prompting_object = EmergentTextSpeech(), eval_data_path=os.path.join(data_dir_path, f"emergent_tts_eval_data.jsonl"))
        os.makedirs(output_dir_path, exist_ok=True)
        prefix = "emergent-tts-eval_strong-prompting" if strong_prompting else "emergent-tts-eval"
        if voice_to_use is not None:
            prefix += f"_{voice_to_use}"
        config.metrics_path = os.path.join(output_dir_path, f"{prefix}_evaluation-metrics.json")
        config.predictions_path = os.path.join(output_dir_path, f"{prefix}_evaluation-predictions.jsonl")
        config.output_audios_dir = os.path.join(output_dir_path, f"{prefix}_output-audios")
        os.makedirs(config.output_audios_dir, exist_ok=True)
        return config

def audio_understanding_model_win_rate(sample: Dict, audio_out_path, client, model_name, task_config, **judger_generation_config) -> None:
    """Score a single sample using judger model"""
    current_audio_array = get_audio_array_from_path(audio_out_path)
    baseline_audio_array = get_audio_array_from_path(sample["baseline_audio_path"])
    audio_1, audio_2 = current_audio_array, baseline_audio_array
    predicted_speech_index = 1
    # Set random seed equal to sample UNOIQUE_KEY_NAME to ensure reproducibility
    seed_temp = sample[UNIQUE_KEY_NAME]
    if np.random.default_rng(seed_temp).random() > 0.5:
        audio_1, audio_2 = baseline_audio_array, current_audio_array
        predicted_speech_index = 2
    system_prompt_judger, user_message, post_audio_1_message = task_config.prompting_object.get_win_rate_prompts(**sample)
    sample["judger_model"] = model_name
    sample["prompt_win_rate_calc"] = user_message
    sample["predicted_speech_index"] = predicted_speech_index
    try:
        resp = client.generate_w_audio_comparison(model_name = model_name, 
                                    system_message = system_prompt_judger, 
                                    user_message = user_message, 
                                    audio_array_1 = audio_1,
                                    post_audio_1_message = post_audio_1_message,
                                    audio_array_2 = audio_2,
                                    **judger_generation_config)
        resp_json = json_repair.loads(resp)
        task_config.prompting_object.validate_win_rate_response(resp_json, resp)
        sample["judger_output_win_rate_based"] = resp_json
        return
    except Exception as e:
        print(f"Error scoring sample from judge_model, {str(e)}")
        sample["judger_output_win_rate_based"] = {"reasoning_system_1": f"Error calling client or parsing response: {str(e)}", 
                                                  "reasoning_system_2": f"Error calling client or parsing response: {str(e)}", 
                                                  "system_comparison": f"Error calling client or parsing response: {str(e)}", 
                                                  "winner": -1}
        return

def get_judge_model_attributes(judge_model, judger_base_url):
    if "gemini" in judge_model:
        client = GeminiClient(api_key=os.getenv("JUDGER_API_KEY"), thinking_budget=256)
    else:
        assert "gpt" in judge_model or judger_base_url is not None, "Please provide gpt as judge_model, or provide a valid judger_base_url to use locally hosted model as judger"
        client = OpenAIClient(api_key=os.getenv("JUDGER_API_KEY"), voice_to_use=None, base_url=judger_base_url)
    generation_config = GENERATION_CONFIG.copy()
    generation_config["temperature"] = 0
    generation_config["max_new_tokens"] = 131072 if isinstance(client, GeminiClient) else 16384
    return client, generation_config

def calculate_exact_metrics(all_results, wer_function):
    """
    Calculates exact metrics like wer, mos score.
    """
    results_dict = defaultdict(list)
    for result in all_results:
        # Calculate WER and CER
        distance, ref_length = wer_function(gt = result["text_to_synthesize"], pred = result["model_prediction"], language = result["language"])
        result["wer"] = distance / ref_length * 100
        results_dict[f"eval/wer"].append(result["wer"])
        category = result["category"]
        category_key = f"eval/{category}_wer"
        results_dict[category_key].append(result["wer"])
        if "evolution_depth" in result:
            evolution_depth = result["evolution_depth"]
            evolution_depth_key = f"eval/{category}_evolution_depth_{evolution_depth}_wer"
            results_dict[evolution_depth_key].append(result["wer"])
        # Calculate MOS
        results_dict[f"eval/mos"].append(result["mos_score"])
    final_result_dict = {}
    for k,v in results_dict.items():
        final_result_dict[k] = np.mean(v) if len(v) > 0 else 0
    final_result_dict["eval/wer_median"] = np.percentile(results_dict[f"eval/wer"], 50) if len(results_dict[f"eval/wer"]) > 0 else 0
    return final_result_dict

def calculate_metrics_win_rate(all_results):
    """Calculate metrics for the given task"""
    results_dict = {
        f"eval/win_rate": [],
        f"eval/tie_rate": []
    }
    failed_count = 0
    for result in all_results:
        if result["judger_output_win_rate_based"]["winner"] != -1:
            if result["judger_output_win_rate_based"]["winner"] == result["predicted_speech_index"]:
                this_sample_win = 1
                this_sample_tie = 0
            # 0 means tie
            elif result["judger_output_win_rate_based"]["winner"] == 0:
                this_sample_win = 0.5
                this_sample_tie = 1
            else:
                this_sample_win = 0
                this_sample_tie = 0
                
            results_dict[f"eval/win_rate"].append(this_sample_win)
            results_dict[f"eval/tie_rate"].append(this_sample_tie)
            # Category
            category = result["category"]
            category_key = f"eval/{category}_win_rate"
            category_key_tie = f"eval/{category}_tie_rate"
            if category_key not in results_dict:
                results_dict[category_key] = []
            if category_key_tie not in results_dict:
                results_dict[category_key_tie] = []
            results_dict[category_key].append(this_sample_win)
            results_dict[category_key_tie].append(this_sample_tie)
            if "evolution_depth" in result:
                evolution_depth = result["evolution_depth"]
                evolution_depth_key = f"eval/{category}_evolution_depth_{evolution_depth}_win_rate"
                evolution_depth_key_tie = f"eval/{category}_evolution_depth_{evolution_depth}_tie_rate"
                if evolution_depth_key not in results_dict:
                    results_dict[evolution_depth_key] = []
                if evolution_depth_key_tie not in results_dict:
                    results_dict[evolution_depth_key_tie] = []
                results_dict[evolution_depth_key].append(this_sample_win)
                results_dict[evolution_depth_key_tie].append(this_sample_tie)
        else:
            failed_count += 1
    win_rate_dict = {"eval/judger_parsing_failed_count": failed_count}
    for k,v in results_dict.items():
        win_rate_dict[k] = np.mean(v) if len(v) > 0 else 0
    return win_rate_dict

def save_predictions(task_config: EmergentTTSEvalConfig, all_results):
    """Save evaluation predictions to JSONL file"""
    with open(task_config.predictions_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def save_metrics_to_json(task_config: EmergentTTSEvalConfig, metrics: Dict):
    """Save evaluation metrics to JSON file"""
    with open(task_config.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def filter_eval_data(eval_data: List[Dict], depths_to_evaluate: List[int], categories_to_evaluate: List[str]):
    prev_length = len(eval_data)
    eval_data = [sample for sample in eval_data if int(sample["evolution_depth"]) in depths_to_evaluate]
    eval_data = [sample for sample in eval_data if sample["category"] in categories_to_evaluate]
    print(f"#IMPORTANT: Removed {prev_length - len(eval_data)} samples from evaluation data AFTER filtering")
    return eval_data

def evaluate_with_accelerate(
    model_client,
    accelerator,
    depths_to_evaluate: str,
    categories_to_evaluate: str,
    seed: int = 42,
    output_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    baseline_audios_path: Optional[str] = None,
    fetch_audios_from_path: Optional[str] = None,
    judge_model: str = "gemini-2.5-pro-preview-03-25",
    temperature: float = 0.0,
    evaluate_function: str = "win_rate",
    strong_prompting: bool = False,
    judger_base_url: Optional[str] = None,
    **kwargs
) -> Optional[Dict]:
    """Evaluate model on local GPU using accelerate"""
    assert accelerator is not None
    if len(evaluate_function) == 0:
        print("Warning: No judger based evaluation will be performed as evaluate_function is empty")
    if len(depths_to_evaluate) == 0 or len(categories_to_evaluate) == 0:
        raise ValueError("depths_to_evaluate or categories_to_evaluate is empty")
    depths_to_evaluate = list(map(int, depths_to_evaluate.split(",")))
    categories_to_evaluate = categories_to_evaluate.split(",")
    evaluate_function_list = evaluate_function.split(",")
    model_generation_config = GENERATION_CONFIG.copy()
    model_generation_config["temperature"] = temperature
    judger_client, judger_generation_config = get_judge_model_attributes(judge_model, judger_base_url)
    model_client.set_eval_mode()
    mosnet_model = Wav2Vec2MOS(os.path.join(DATA_BASE_PATH, "wv_mos.ckpt")).to(accelerator.device)
    start_time = time.time()
    set_seed(seed)
    voice_to_use = None
    if hasattr(model_client, "voice_to_use"):
        voice_to_use = model_client.voice_to_use
    # TODO: Allow custom data_base_path
    task_config = EmergentTTSEvalConfig.create(DATA_BASE_PATH, output_dir, strong_prompting, voice_to_use)
    # TODO: Uncomment this
    # if calculate_file_md5(task_config.eval_data_path) != md5_hash_value:
    #     raise ValueError(f"md5 mismatch for {task_config.eval_data_path}, please check the data")
    
    with open(task_config.eval_data_path, 'r', encoding='utf-8') as source:
        eval_data = []
        for line in source:
            eval_data.append(json.loads(line))
        eval_data = add_unique_id_to_samples(eval_data)
        eval_data = filter_eval_data(eval_data, depths_to_evaluate, categories_to_evaluate)
    # Randomly select num_samples items from eval_data list
    if num_samples is not None:
        eval_data = np.random.default_rng(seed=seed).choice(eval_data, min(num_samples, len(eval_data)), replace=False).tolist()

    print(f"Running evaluation on {len(eval_data)} samples")
    if fetch_audios_from_path is not None:
        print(f"#INFO: Fetching audios from {fetch_audios_from_path}, not generating audios from model")

    task_language = eval_data[0]["language"]
    task_class, model_path = source_language_to_hf_transcriber[task_language]
    transcriber_model = task_class(model_path, accelerator.device)
    sharded_results = []
    with accelerator.split_between_processes(eval_data) as sharded_eval_data:
        with torch.no_grad():
            for sample in tqdm(sharded_eval_data, desc=f"Evaluating on EmergentTTS-Eval"):
                try:
                    audio_out_path = os.path.join(task_config.output_audios_dir, f"{sample[UNIQUE_KEY_NAME]}.wav")
                    system_message, user_message = model_client.prepare_emergent_tts_sample(**sample, strong_prompting=strong_prompting, prompting_object=task_config.prompting_object)
                    sample["system_message"] = system_message
                    sample["user_message"] = user_message
                    if fetch_audios_from_path is None:
                        response_pydub_segment, _ = model_client.generate_audio_out(system_message, user_message, accelerator, **model_generation_config)
                        response_pydub_segment.export(audio_out_path, format="wav") # Save the audio to the output path
                    else:
                        fetch_path = os.path.join(fetch_audios_from_path, f"{sample[UNIQUE_KEY_NAME]}.wav")
                        assert os.path.exists(fetch_path), f"Audio not found for {sample[UNIQUE_KEY_NAME]}.wav at {fetch_path}"
                        response_pydub_segment = AudioSegment.from_wav(fetch_path)
                        audio_out_path = fetch_path
                    output_transcription = transcriber_model.transcribe(response_pydub_segment)
                    if "win_rate" in evaluate_function_list:
                        assert baseline_audios_path is not None, "Baseline audios path is required to compute win-rate"
                        unique_id_eval = sample[UNIQUE_KEY_NAME]
                        baseline_audio_path = os.path.join(baseline_audios_path, f"{unique_id_eval}.wav")
                        assert os.path.exists(baseline_audio_path), f"Baseline audio not found for {unique_id_eval}.wav"
                        # Convert to absolute path
                        baseline_audio_path = os.path.abspath(baseline_audio_path)
                        sample["baseline_audio_path"] = baseline_audio_path
                        audio_understanding_model_win_rate(sample, audio_out_path, judger_client, judge_model, task_config, **judger_generation_config)
                    mos_score = mosnet_model.calculate_one(audio_out_path, accelerator.device)
                    result = {
                        **sample,
                        "model_prediction": output_transcription,
                        "audio_out_path": audio_out_path,
                        "mos_score": mos_score
                    }
                    sharded_results.append(result)
                except Exception as e:
                    print(f"Error processing sample {sample[UNIQUE_KEY_NAME]}: {str(e)}")
                    sharded_results.append(None)
                    traceback.print_exc()
                    continue

    torch.cuda.empty_cache()
    all_results = accelerator.gather_for_metrics(sharded_results)
    # Remove none values and set as skipped_samples
    skipped_samples = all_results.count(None)
    all_results = [r for r in all_results if r is not None]
    all_results = get_dedup_results(all_results)

    with AccelerateSyncBlock(accelerator):
        if accelerator.is_main_process:
            try:
                win_rate_dict = {}
                if "win_rate" in evaluate_function_list:
                    win_rate_dict = calculate_metrics_win_rate(all_results)
                # Calculate other metrics like WER, CER, MOS score.
                final_result_dict = calculate_exact_metrics(all_results, calculate_wer_qwen)
                final_result_dict.update(win_rate_dict)
                final_result_dict["voice_to_use"] = voice_to_use
                final_result_dict["skipped_samples"] = skipped_samples
                # Sort result_dict by keys
                final_result_dict = dict(sorted(final_result_dict.items()))
                print(f"EmergentTTS-Eval evaluation results: {final_result_dict}")
                save_predictions(task_config, all_results)
                save_metrics_to_json(task_config, final_result_dict)
                print(f"Total time taken for {time.time() - start_time}, skipped samples: {skipped_samples}")
            except Exception as e:
                print(f"Error during EmergentTTS-Eval metrics calculation: {str(e)}")
                traceback.print_exc()
    
def process_sample(sample, model_client, model_name, task_config, evaluate_function_list, judge_model, judger_base_url, strong_prompting, helper_model_dict, baseline_audios_path, fetch_audios_from_path, **model_generation_config):
    judger_client, judger_generation_config = get_judge_model_attributes(judge_model, judger_base_url)
    helper_model_idx = sample[UNIQUE_KEY_NAME] % len(helper_model_dict["transcriber_clients"])
    with helper_model_dict["model_locks"][helper_model_idx]:
        transcriber = helper_model_dict["transcriber_clients"][helper_model_idx]
        mosnet_model = helper_model_dict["mosnet_models"][helper_model_idx]
        device_to_use = helper_model_dict["devices_to_use"][helper_model_idx]
        try:
            audio_out_path = os.path.join(task_config.output_audios_dir, f"{sample[UNIQUE_KEY_NAME]}.wav")
            system_message, user_message = model_client.prepare_emergent_tts_sample(**sample, strong_prompting=strong_prompting, prompting_object=task_config.prompting_object)
            sample["system_message"] = system_message
            sample["user_message"] = user_message
            if fetch_audios_from_path is None:
                response_pydub_segment, _ = model_client.generate_audio_out(
                    model_name, system_message, user_message, **model_generation_config
                )
                response_pydub_segment.export(audio_out_path, format="wav")
            else:
                fetch_path = os.path.join(fetch_audios_from_path, f"{sample[UNIQUE_KEY_NAME]}.wav")
                assert os.path.exists(fetch_path), f"Audio not found for {sample[UNIQUE_KEY_NAME]}.wav at {fetch_path}"
                response_pydub_segment = AudioSegment.from_wav(fetch_path)
                audio_out_path = fetch_path
            output_transcription = transcriber.transcribe(response_pydub_segment)
            if "win_rate" in evaluate_function_list:
                assert baseline_audios_path is not None, "Baseline audios path is required to compute win-rate"
                unique_id_eval = sample[UNIQUE_KEY_NAME]
                baseline_audio_path = os.path.join(baseline_audios_path, f"{unique_id_eval}.wav")
                assert os.path.exists(baseline_audio_path), f"Baseline audio not found for {unique_id_eval}.wav"
                # Convert to absolute path
                baseline_audio_path = os.path.abspath(baseline_audio_path)
                sample["baseline_audio_path"] = baseline_audio_path
                audio_understanding_model_win_rate(sample, audio_out_path, judger_client, judge_model, task_config, **judger_generation_config)
            mos_score = mosnet_model.calculate_one(audio_out_path, device_to_use)
            result = {
                **sample,
                "model_prediction": output_transcription,
                "audio_out_path": audio_out_path,
                "mos_score": mos_score
            }
            return result
        except Exception as e:
            print(f"Error processing sample {sample[UNIQUE_KEY_NAME]} through API: {str(e)}")
            return None

def eval_api_closed_model(
    model_client,
    accelerator,
    depths_to_evaluate: str,
    categories_to_evaluate: str,
    seed: int = 42,
    output_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    baseline_audios_path: Optional[str] = None,
    fetch_audios_from_path: Optional[str] = None,
    judge_model: str = "gemini-2.5-pro-preview-03-25",
    temperature: float = 0.0,
    evaluate_function: str = "win_rate",
    strong_prompting: bool = False,
    judger_base_url: Optional[str] = None,
    num_threads: int = 1,
    model_name: str = "",
    ):
    assert accelerator is None
    if len(evaluate_function) == 0:
        print("Warning: No judger based evaluation will be performed as evaluate_function is empty")
    if len(depths_to_evaluate) == 0 or len(categories_to_evaluate) == 0   :
        raise ValueError("depths_to_evaluate or categories_to_evaluate is empty")
    depths_to_evaluate = list(map(int, depths_to_evaluate.split(",")))
    categories_to_evaluate = categories_to_evaluate.split(",")
    evaluate_function_list = evaluate_function.split(",")
    assert len(evaluate_function_list) != 0
    model_generation_config = GENERATION_CONFIG.copy()
    model_generation_config["temperature"] = temperature
    start_time = time.time()
    voice_to_use = None
    if hasattr(model_client, "voice_to_use"):
        voice_to_use = model_client.voice_to_use
    task_config = EmergentTTSEvalConfig.create(DATA_BASE_PATH, output_dir, strong_prompting, voice_to_use)

    # TODO: Uncomment this
    # if calculate_file_md5(task_config.eval_data_path) != md5_hash_value:
    #     raise ValueError(f"md5 mismatch for {task_config.eval_data_path}, please check the data")
    
    with open(task_config.eval_data_path, 'r', encoding='utf-8') as source:
        eval_data = []
        for line in source:
            eval_data.append(json.loads(line))
        eval_data = add_unique_id_to_samples(eval_data)
        eval_data = filter_eval_data(eval_data, depths_to_evaluate, categories_to_evaluate)
 
    if num_samples is not None:
        eval_data = np.random.default_rng(seed=seed).choice(eval_data, min(num_samples, len(eval_data)), replace=False).tolist()

    print(f"Running evaluation on {len(eval_data)} samples")
    if fetch_audios_from_path is not None:
        print(f"#INFO: Fetching audios from {fetch_audios_from_path}, not generating audios from model")

    task_langauge = eval_data[0]["language"]
    # Create devices list, based on all available GPUs. To limit usage, pass CUDA_VISIBLE_DEVICES environment variable
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    num_devices = len(devices)
    print(f"#INFO: Using {num_devices} GPU devices for evaluation and {num_threads} threads")
    # Create num_threads instances of transcriber_client and mosnet_model, place them in thread_idx%num_devices
    print("INFO: Initializing transcriber and mosnet models")
    transcriber_class, model_path = source_language_to_hf_transcriber[task_langauge]
    devices_to_use = [devices[idx % num_devices] for idx in range(num_threads)]
    transcriber_clients = [transcriber_class(model_path, devices_to_use[idx]) for idx in range(num_threads)]
    mosnet_models = [Wav2Vec2MOS(os.path.join(DATA_BASE_PATH, "wv_mos.ckpt")).to(devices_to_use[idx]) for idx in range(num_threads)]
    model_locks = [Lock() for _ in range(num_threads)]
    helper_model_dict = {
        "transcriber_clients": transcriber_clients,
        "mosnet_models": mosnet_models,
        "devices_to_use": devices_to_use,
        "model_locks": model_locks
    }
    print("INFO: Starting evaluation")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        func = partial(process_sample, model_client=model_client, model_name=model_name, task_config=task_config, evaluate_function_list=evaluate_function_list, judge_model=judge_model, judger_base_url=judger_base_url, strong_prompting=strong_prompting, helper_model_dict = helper_model_dict, baseline_audios_path=baseline_audios_path, fetch_audios_from_path=fetch_audios_from_path, **model_generation_config)
        results = list(tqdm(executor.map(func, eval_data), total=len(eval_data)))
        # Filter out None results
        skipped_samples = results.count(None)
        results = [r for r in results if r is not None]

    win_rate_dict = {}
    if "win_rate" in evaluate_function_list:
        win_rate_dict = calculate_metrics_win_rate(results)
    final_result_dict = calculate_exact_metrics(results, calculate_wer_qwen)
    final_result_dict.update(win_rate_dict)
    final_result_dict["voice_to_use"] = voice_to_use
    final_result_dict["skipped_samples"] = skipped_samples
    # Sort result_dict by keys
    final_result_dict = dict(sorted(final_result_dict.items()))
    print(f"EmergentTTS-Eval evaluation results: {final_result_dict}")
    save_predictions(task_config, results)
    save_metrics_to_json(task_config, final_result_dict)
    print(f"Total time taken for {time.time() - start_time}, skipped samples: {skipped_samples}")