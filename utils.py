from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperModel, WhisperProcessor

from config import LLAMA_MODEL_NAME, WHISPER_MODEL_NAME


def load_processors_and_models():
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)

    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME)

    return whisper_processor, whisper_model, llama_tokenizer, llama_model
