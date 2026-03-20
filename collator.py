from config import MAX_TEXT_LEN, PROMPT_TEXT, SAMPLE_RATE, USE_PROMPT


class SpeechCollator:
    def __init__(self, whisper_processor, llama_tokenizer):
        self.whisper_processor = whisper_processor
        self.llama_tokenizer = llama_tokenizer

    def __call__(self, batch):
        audio_arrays = [x["audio"]["array"] for x in batch]
        texts = [x["text"] for x in batch]

        whisper_inputs = self.whisper_processor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        if USE_PROMPT:
            prompt_texts = [PROMPT_TEXT + t for t in texts]
        else:
            prompt_texts = texts

        tokenized = self.llama_tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )

        return {
            "input_features": whisper_inputs["input_features"],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "raw_texts": texts,
        }
