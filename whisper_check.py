from transformers import WhisperModel

m = WhisperModel.from_pretrained("openai/whisper-base")
print(m.config.d_model)

import torch
from transformers import WhisperModel, AutoModelForCausalLM

whisper = WhisperModel.from_pretrained("openai/whisper-base")
llama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

d_whisper = whisper.config.d_model
d_llama = llama.config.hidden_size

print("Whisper dim:", d_whisper)
print("Llama dim:", d_llama)

projector = torch.nn.Linear(d_whisper, d_llama)

dummy = torch.randn(2, 100, d_whisper)
out = projector(dummy)

print("Projected shape:", out.shape)