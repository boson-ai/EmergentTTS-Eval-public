import torch
from pydub import AudioSegment
import numpy as np

class Qwen2_5OmniClient:
    def __init__(self, model_name, accelerator, voice_to_use):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=accelerator.device,
            torch_dtype="auto",
            enable_audio_output=True,
            attn_implementation="flash_attention_2",
        )

        self.sampling_rate = 24000
        self.voice_to_use = voice_to_use if voice_to_use is not None else "Chelsie"
        assert self.voice_to_use in ["Chelsie","Ethan"]

    def set_eval_mode(self):
        self.model.eval()

    def generate_audio_out(self, system_message, user_message, accelerator, **GENERATION_CONFIG):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=text_prompt,
            return_tensors="pt",
            padding=True
        )
        for k,v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device=accelerator.device)
        _, audio_wv = self.model.generate(
                **inputs,
                return_audio=True,
                thinker_max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                talker_do_sample=True,
                talker_temperature=GENERATION_CONFIG["temperature"],
                talker_top_p=GENERATION_CONFIG["top_p"],
                talker_max_new_tokens=GENERATION_CONFIG["max_audio_new_tokens"],
                speaker=self.voice_to_use
            )
        audio_np = audio_wv.cpu().numpy()

        # Normalize to 16-bit PCM format
        audio_np = np.int16(audio_np * 32767)

        # Create an audio segment
        pydub_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=self.sampling_rate,
            sample_width=2,  # 16-bit audio
            channels=1
        )
        return pydub_segment, _
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        qwen_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        if strong_prompting:
            user_message = prompting_object.USER_MESSAGE_STRONG_TEMPLATE.replace("{{{descriptions}}}", prompting_object.ALL_DESCRIPTIONS[category]).replace("{{{text_to_synthesize}}}", text_to_synthesize)
        else:
            user_message = prompting_object.USER_MESSAGE_DEFAULT_TEMPLATE.replace("{{{text_to_synthesize}}}", text_to_synthesize)
        return qwen_system_prompt, user_message

class Sesame1BClient:
    def __init__(self, model_name, accelerator):
        from generator import load_csm_1b
        import torchaudio
        self.model = load_csm_1b(device=accelerator.device)
        self.sampling_rate = self.model.sample_rate

    def set_eval_mode(self):
        pass

    def generate_audio_out(self, system_message, user_message, accelerator, **GENERATION_CONFIG):
        out_audio = self.model.generate(
            text=user_message,
            speaker=0,
            context=[],
            max_audio_length_ms=120_000,
            temperature=GENERATION_CONFIG["temperature"],
        )
        out_audio = out_audio.cpu().numpy()
        #Convert to 16-bit PCM format
        out_audio = np.int16(out_audio * 32767)
        #Create an audio segment
        pydub_segment = AudioSegment(
            out_audio.tobytes(),
            frame_rate=self.sampling_rate,
            sample_width=2,
            channels=1
        )
        return pydub_segment, None
    
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        return None, text_to_synthesize


class OrpheusClient:
    def __init__(self, model_name, accelerator, voice_to_use):
        from snac import SNAC
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model.to(accelerator.device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=accelerator.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.voice_to_use = voice_to_use if voice_to_use is not None else "tara"
        self.sampling_rate = 24000
        self.device = accelerator.device
    def set_eval_mode(self):
        pass

    def generate_audio_out(self, system_message, user_message, accelerator, **GENERATION_CONFIG):
        prompts = [user_message]
        prompts = [f"{self.voice_to_use}: " + p for p in prompts]
        all_input_ids = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)
        start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human
        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)
        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        input_ids = all_padded_tensors.to(self.device)
        attention_mask = all_attention_masks.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=GENERATION_CONFIG["max_audio_new_tokens"],
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )
        #@title Parse Output as speech
        token_to_find = 128257
        token_to_remove = 128258

        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []

        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []

        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)


        def redistribute_codes(code_list):
            layer_1 = []
            layer_2 = []
            layer_3 = []
            for i in range((len(code_list)+1)//7):
                layer_1.append(code_list[7*i])
                layer_2.append(code_list[7*i+1]-4096)
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_2.append(code_list[7*i+4]-(4*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
            codes = [torch.tensor(layer_1).unsqueeze(0).to(self.device),
                    torch.tensor(layer_2).unsqueeze(0).to(self.device),
                    torch.tensor(layer_3).unsqueeze(0).to(self.device)]
            audio_hat = self.snac_model.decode(codes)
            return audio_hat

        my_samples = []
        for code_list in code_lists:
            samples = redistribute_codes(code_list)
            my_samples.append(samples)

        assert len(my_samples) == 1
        out_array = my_samples[0].detach().squeeze().to("cpu").numpy()
        # Normalize to 16-bit PCM format
        out_array = np.int16(out_array * 32767)
        # Create an audio segment
        pydub_segment = AudioSegment(
            out_array.tobytes(),
            frame_rate=self.sampling_rate,
            sample_width=2,
            channels=1
        )
        return pydub_segment, None
    
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        return None, text_to_synthesize