import os
# TODO: Remove this import
if not os.getenv("DISABLE_HIGGS_AUDIO_MODEL"):
    from boson_multimodal.model.higgs_audio import HiggsAudioModel
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
import torch
from accelerate import Accelerator

from model_clients import (
    HiggsAudioClient, # TODO: Remove this import
    MiniCPMClient,
    KimiClient,
    Qwen2_5OmniClient,
    Sesame1BClient,
    BarkClient,
    OrpheusClient
)
from api_clients import (
    OpenAIClient,
    OpenAIDirectClient,
    ElevenLabsClient,
    DeepgramClient,
    HumeAIClient
)
from inference import evaluate_with_accelerate, eval_api_closed_model

## Model name to client mapping for HF models, please add new model here along with client function.
hf_model_name_to_client = {
    "openbmb/MiniCPM-o-2_6": MiniCPMClient,
    "moonshotai/Kimi-Audio-7B-Instruct": KimiClient,
    "Qwen/Qwen2.5-Omni-7B": Qwen2_5OmniClient,
    "suno/bark": BarkClient,
    "canopylabs/orpheus-tts-0.1-finetune-prod": OrpheusClient
}
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint or model name")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate on, by default all samples are used")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random torch seed for reproducibility with non-api models")
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        type=str,
        choices=["sdpa", "flash_attention_2", "eager"],
        help="The implementation of the attention layer.",
    )
    # Benchmark specific arguments
    parser.add_argument(
        "--judge_model_provider",
        type=str,
        default="gemini-2.5-pro-preview-03-25",
        help="Model provider for model-as-judge evaluation",
    )
    parser.add_argument(
        "--api_num_threads",
        type=int,
        default=4,
        help="Number of parallel threads to use for api calls for api models",
    )
    parser.add_argument(
        "--judger_base_url",
        type=str,
        default=None,
        help="Base url for VLLM API, if judger model is hosted locally through openai compatible api",
    )
    ## TODO: Remove this argument
    parser.add_argument(
        "--audio_tokenizer_type",
        type=str,
        default=None,
        help="Audio tokenizer to use for the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use for benchmarking audio out.",
    )
    parser.add_argument(
        "--tts_judger_evaluate_function",
        type=str,
        default="",
        help="Pass win_rate to compute win rate, otherwise keep empty",
    )
    parser.add_argument(
        "--text_speech_strong_prompting",
        action="store_true",
        help="If set, will use strong prompting for text_speech_eval",
    )
    parser.add_argument(
        "--depths_to_evaluate",
        type=str,
        default="0,1,2,3",
        help="Comma separated list of evolution depths to evaluate, by default all depths are evaluated",
    )
    parser.add_argument(
        "--categories_to_evaluate",
        type=str,
        default="Emotions,Paralinguistics,Syntactic Complexity,Foreign Words,Questions,Pronunciation",
        help="Comma separated list of categories to evaluate, by default all categories are evaluated",
    )
    parser.add_argument(
        "--voice_to_use",
        type=str,
        default=None,
        help="The voice to use for any specific client, make sure the client supports this, if not passed, a default voice will be used",
    )
    parser.add_argument(
        "--baseline_audios_path",
        type=str,
        default=None,
        help="Path to baseline audios, if not passed, win-rate cannot be computed",
    )
    parser.add_argument(
        "--fetch_audios_from_path",
        type=str,
        default=None,
        help="Path to fetch audios from, if passed, the model will not be asked to generate audios, instead the audios from the given path will be used for judger evaluation",
    )
    args = parser.parse_args()
    # Print parser args in a nice way
    print("-----------------------------------")
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-----------------------------------")

    # Prepare kwargs
    kwargs = {
        "judge_model": args.judge_model_provider, 
        "temperature": args.temperature, 
        "evaluate_function": args.tts_judger_evaluate_function,
        "strong_prompting": args.text_speech_strong_prompting,
        "judger_base_url": args.judger_base_url
    }

    # Check if model path exists locally
    # TODO: Remove this block
    accelerator = None
    if os.path.exists(args.model_name_or_path):
        print("Evaluating local HiggsAudio model")
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600000))]
        accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
        # Load model, tokenizer and config
        model = HiggsAudioModel.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        whisper_processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            trust_remote=True,
        )
        model_client = HiggsAudioClient(model, tokenizer, config, whisper_processor, accelerator, args.audio_tokenizer_type, args.seed)
        evaluate_package = evaluate_with_accelerate
    elif args.model_name_or_path in hf_model_name_to_client:
        print(f"Evaluating HF hub model {args.model_name_or_path}")
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600000))]
        accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
        model_client = hf_model_name_to_client[args.model_name_or_path](args.model_name_or_path, accelerator, args.voice_to_use)
        evaluate_package = evaluate_with_accelerate
    elif args.model_name_or_path == "Sesame1B":
        print(f"Evaluating Sesame1B model, make sure it is in PYTHONPATH")
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600000))]
        accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
        model_client = Sesame1BClient(args.model_name_or_path, accelerator)
        evaluate_package = evaluate_with_accelerate
    else:
        print("Evaluating closed source model through api client")
        if "gpt" in args.model_name_or_path:
            if "tts" in args.model_name_or_path:
                model_client = OpenAIDirectClient(api_key=os.getenv("OPENAI_API_KEY"), voice_to_use=args.voice_to_use)
            else:
                model_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), voice_to_use=args.voice_to_use)
        elif "eleven" in args.model_name_or_path:
            model_client = ElevenLabsClient(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_to_use=args.voice_to_use)
        elif "deepgram" in args.model_name_or_path:
            model_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"), voice_to_use=args.voice_to_use)
        elif "Hume" in args.model_name_or_path:
            model_client = HumeAIClient(api_key=os.getenv("HUME_API_KEY"))
        else:
            raise NotImplementedError(f"Model {args.model_name_or_path} not supported")
        kwargs["num_threads"] = args.api_num_threads
        kwargs["model_name"] = args.model_name_or_path
        evaluate_package = eval_api_closed_model
    evaluate_package(
            model_client=model_client,
            accelerator=accelerator,
            depths_to_evaluate=args.depths_to_evaluate,
            categories_to_evaluate=args.categories_to_evaluate,
            seed=args.seed,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            baseline_audios_path=args.baseline_audios_path,
            fetch_audios_from_path=args.fetch_audios_from_path,
            **kwargs,
        )