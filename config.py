WHISPER_MODEL_NAME = "openai/whisper-base"
LLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SAMPLE_RATE = 16000
MAX_TEXT_LEN = 128


TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LR = 2e-5
NUM_EPOCHS = 8

USE_PROMPT = True
PROMPT_TEXT = "Transcribe this audio: "

DEVICE = "cuda"
