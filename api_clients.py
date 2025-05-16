import openai
import base64
from tenacity import retry, wait_fixed, stop_after_attempt
from utils_eval import get_wav_in_memory, TARGET_RATE
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydub import AudioSegment
from io import BytesIO

class OpenAIClient():
    def __init__(self, api_key, voice_to_use, base_url=None):
        if base_url is not None:
            self.client = openai.Client(api_key=api_key, base_url=base_url)
        else:
            self.client = openai.Client(api_key=api_key)
        self.voice_to_use = voice_to_use if voice_to_use is not None else "alloy"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generate_audio_out(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        try:
            user_content = [user_message]
            completion = self.client.chat.completions.create(
                model=model_name,
                modalities=["text", "audio"],
                audio={"voice": self.voice_to_use, "format": "wav"},
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                max_tokens = GENERATION_CONFIG["max_audio_new_tokens"],
                temperature = GENERATION_CONFIG["temperature"],
                top_p = GENERATION_CONFIG["top_p"],
            )
            completion_audio = completion.choices[0].message.audio
            wav_bytes, transcript = base64.b64decode(completion_audio.data), completion_audio.transcript
            pydub_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
            return pydub_segment, transcript
        
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e
        
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        if strong_prompting:
            user_message = prompting_object.USER_MESSAGE_STRONG_TEMPLATE.replace("{{{descriptions}}}", prompting_object.ALL_DESCRIPTIONS[category]).replace("{{{text_to_synthesize}}}", text_to_synthesize)
        else:
            user_message = prompting_object.USER_MESSAGE_DEFAULT_TEMPLATE.replace("{{{text_to_synthesize}}}", text_to_synthesize)
        return prompting_object.SYSTEM_PROMPT_DEFAULT, user_message
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generate_w_audio_comparison(self, model_name, system_message, user_message, audio_array_1, post_audio_1_message, audio_array_2, **GENERATION_CONFIG):
        try:
            audio_bytes_1 = get_wav_in_memory(audio_array_1)
            encoded_audio_string_1 = base64.b64encode(audio_bytes_1).decode("utf-8")

            audio_bytes_2 = get_wav_in_memory(audio_array_2)
            encoded_audio_string_2 = base64.b64encode(audio_bytes_2).decode("utf-8")

            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {"type": "input_audio", "input_audio": {"data": encoded_audio_string_1, "format": "wav"}},
                            {"type": "text", "text": post_audio_1_message},
                            {"type": "input_audio", "input_audio": {"data": encoded_audio_string_2, "format": "wav"}},
                        ],
                    },
                ],
                max_tokens = GENERATION_CONFIG["max_new_tokens"],
                temperature = GENERATION_CONFIG["temperature"],
                top_p = GENERATION_CONFIG["top_p"],
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e


class OpenAIDirectClient():
    def __init__(self, api_key, voice_to_use):
        self.client = openai.Client(api_key=api_key)
        self.voice_to_use = voice_to_use if voice_to_use is not None else "alloy"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generate_audio_out(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        try:
            response = openai.audio.speech.create(
                model=model_name,
                voice=self.voice_to_use,
                input=user_message,
                instructions=system_message,
                response_format="wav"
            )
            audio_buffer = BytesIO()
            for chunk in response.iter_bytes():
                audio_buffer.write(chunk)

            # Seek to the start of the buffer
            audio_buffer.seek(0)

            # Load into pydub from memory
            audio_segment = AudioSegment.from_file(audio_buffer, format="wav")
            return audio_segment, None
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e
    
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        if strong_prompting:
            system_message = prompting_object.ALL_DESCRIPTIONS[category]
        else:
            system_message = ""
        return system_message, text_to_synthesize

class GeminiClient():
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    @retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
    def generate_w_audio_comparison(self, model_name, system_message, user_message, audio_array_1, post_audio_1_message, audio_array_2, **GENERATION_CONFIG):
        try:
            generation_config = genai.GenerationConfig(max_output_tokens = GENERATION_CONFIG["max_new_tokens"],
                                                       temperature = GENERATION_CONFIG["temperature"],
                                                       top_p = GENERATION_CONFIG["top_p"])
            wav_audio_bytes_1 = get_wav_in_memory(audio_array_1)
            wav_audio_bytes_2 = get_wav_in_memory(audio_array_2)
            request_payload = [user_message, {"mime_type": f"audio/wav", "data": wav_audio_bytes_1}, post_audio_1_message, {"mime_type": f"audio/wav", "data": wav_audio_bytes_2}]

            model = genai.GenerativeModel(model_name, system_instruction=system_message)
            response = model.generate_content(
                request_payload,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
            )
            return response.text
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e

class ElevenLabsClient():
    def __init__(self, api_key, voice_to_use):
        from elevenlabs import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        self.voice_to_use = voice_to_use if voice_to_use is not None else "JBFqnCBsd6RMkjVDRZzb"
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def generate_audio_out_prompt_tts(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        """
        In this case, system message contains the style prompt and user_message contains the content.
        """
        try:
            out_audio_string = self.client.text_to_voice.create_previews(
                        voice_description=system_message,
                        text=user_message,
                    ).previews[0].audio_base_64
            out_audio_bytes = base64.b64decode(out_audio_string)
            mp3_buffer = BytesIO(out_audio_bytes)
            mp3_buffer.seek(0)
            pydub_segment = AudioSegment.from_file(
                mp3_buffer, format="mp3"
            )
            pydub_segment = pydub_segment.set_channels(1).set_sample_width(2)
            return pydub_segment, None
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e
        
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def generate_audio_out(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        """
        In this case, the user message contains the text to be synthesized.
        """
        try:
            out_audio = self.client.text_to_speech.convert(
                voice_id=self.voice_to_use,
                output_format="mp3_22050_32",
                text=user_message,
                model_id=model_name,
            )
            mp3_buffer = BytesIO()
            for chunk in out_audio:
                mp3_buffer.write(chunk)
            # Convert bytes to io.BytesIO
            mp3_buffer.seek(0)
            pydub_segment = AudioSegment.from_file(
                mp3_buffer, format="mp3"
            )
            pydub_segment = pydub_segment.set_channels(1).set_sample_width(2)
            return pydub_segment, None
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e
    
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        return None, text_to_synthesize
        
class DeepgramClient():
    def __init__(self, api_key, voice_to_use):
        import requests
        from functools import partial
        self.sampling_rate = TARGET_RATE
        url = "https://api.deepgram.com/v1/speak"
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        self.voice_to_use = voice_to_use if voice_to_use is not None else "amalthea-en"
        query_params = {
            "encoding": "mp3",
            "model": f"aura-2-{self.voice_to_use}"
        }
        self.client = partial(requests.post, url, headers=headers, params=query_params)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generate_audio_out(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        try:
            response = self.client(json={"text": user_message})
            mp3_bytes = response.content
            mp3_io = BytesIO(mp3_bytes)
            # Load with pydub
            audio_segment = AudioSegment.from_file(mp3_io, format="mp3")
            return audio_segment, None
        except Exception as e:
            print(f"Retrying after error: {str(e)}")
            raise e
    
    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        return None, text_to_synthesize

class HumeAIClient():
    def __init__(self, api_key):
        from hume.client import HumeClient
        self.client = HumeClient(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def generate_audio_out(self, model_name, system_message, user_message, **GENERATION_CONFIG):
        from hume.tts import PostedUtterance
        from hume.tts.types.format_wav import FormatWav
        speech = self.client.tts.synthesize_json(
            utterances=[
                PostedUtterance(
                    description=system_message,
                    text=user_message,
                )
            ],
            format=FormatWav(),
        )
        # Speech contains the wav string, create pydub segment
        b64_wav_str = speech.generations[0].audio
        audio_bytes = base64.b64decode(b64_wav_str)
        pydub_segment = AudioSegment.from_wav(BytesIO(audio_bytes))
        return pydub_segment, None

    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object,**kwargs):
        if strong_prompting:
            system_message = prompting_object.ALL_DESCRIPTIONS[category]
        else:
            system_message = ""
        return system_message, text_to_synthesize