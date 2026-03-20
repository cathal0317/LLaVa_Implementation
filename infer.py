import argparse
import os
from datetime import datetime
import torch
import torchaudio
import numpy as np

from config import DEVICE, PROMPT_TEXT, SAMPLE_RATE, USE_PROMPT
from dataset import load_asr_dataset
from model import SpeechToLlamaModel
from utils import load_processors_and_models


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)


def build_inputs(whisper_processor, llama_tokenizer, audio_tensor, prompt_text):
    if isinstance(audio_tensor, torch.Tensor):
        audio_array = audio_tensor.numpy()
    elif isinstance(audio_tensor, np.ndarray):
        audio_array = audio_tensor
    else:
        audio_array = np.asarray(audio_tensor)
    whisper_inputs = whisper_processor(
        [audio_array],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    tokenized = llama_tokenizer(
        [prompt_text],
        return_tensors="pt",
    )

    return whisper_inputs["input_features"], tokenized["input_ids"], tokenized["attention_mask"]


def save_output(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)


def find_latest_checkpoint(checkpoints_dir="checkpoints"):
    if not os.path.isdir(checkpoints_dir):
        return None
    candidates = [
        os.path.join(checkpoints_dir, name)
        for name in os.listdir(checkpoints_dir)
        if name.endswith(".pt")
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", default=None, help="Path to wav/flac/mp3 file")
    parser.add_argument("--dataset_split", default="validation", help="HF split if no audio_path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index for dataset")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--output_path", default=None, help="Save decoded text to file")
    args = parser.parse_args()

    whisper_processor, whisper_model, llama_tokenizer, llama_model = load_processors_and_models()

    model = SpeechToLlamaModel(
        whisper_model=whisper_model,
        llama_model=llama_model,
        freeze_whisper=True,
        freeze_llama=True,
    ).to(DEVICE)

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.eval()

    if args.audio_path:
        audio = load_audio(args.audio_path)
        reference_text = None
        source_desc = args.audio_path
    else:
        ds = load_asr_dataset(split=args.dataset_split, max_samples=args.sample_idx + 1)
        sample = ds[args.sample_idx]
        audio = sample["audio"]["array"]
        reference_text = sample["text"]
        source_desc = f"{args.dataset_split}[{args.sample_idx}]"
    
    prompt_text = PROMPT_TEXT if USE_PROMPT else ""
    input_features, input_ids, attention_mask = build_inputs(
        whisper_processor,
        llama_tokenizer,
        audio,
        prompt_text,
    )

    input_features = input_features.to(DEVICE)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        encoder_outputs = model.whisper.encoder(input_features=input_features)
        speech_hidden = encoder_outputs.last_hidden_state

        # Match training downsampling (stride=4)
        speech_hidden = speech_hidden.transpose(1, 2)
        speech_hidden = torch.nn.functional.avg_pool1d(
            speech_hidden,
            kernel_size=4,
            stride=4,
        )
        speech_hidden = speech_hidden.transpose(1, 2)

        speech_embeds = model.projector(speech_hidden)

        text_embeds = model.llama.get_input_embeddings()(input_ids)
        if speech_embeds.dtype != text_embeds.dtype:
            speech_embeds = speech_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([speech_embeds, text_embeds], dim=1)

        speech_mask = torch.ones(
            speech_embeds.shape[0],
            speech_embeds.shape[1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([speech_mask, attention_mask], dim=1)

        outputs = model.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            eos_token_id=llama_tokenizer.eos_token_id,
            pad_token_id=llama_tokenizer.eos_token_id,
        )

    decoded = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result_text = decoded[0]

    output_lines = [
        f"source: {source_desc}",
        f"prompt: {prompt_text}",
        f"prediction: {result_text}",
    ]
    if reference_text is not None:
        output_lines.insert(2, f"reference: {reference_text}")

    payload = "\n".join(output_lines) + "\n"
    print(payload)

    if args.output_path:
        save_output(args.output_path, payload)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_output(os.path.join("outputs", f"infer_{ts}.txt"), payload)


if __name__ == "__main__":
    main()
