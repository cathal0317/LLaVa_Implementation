import argparse

import evaluate
import torch
import torch.nn.functional as F

from config import DEVICE, PROMPT_TEXT, SAMPLE_RATE, USE_PROMPT
from dataset import load_asr_dataset
from model import SpeechToLlamaModel
from utils import load_processors_and_models


def build_batch_inputs(whisper_processor, llama_tokenizer, audio_arrays, prompt_text):
    whisper_inputs = whisper_processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    tokenized = llama_tokenizer(
        [prompt_text] * len(audio_arrays),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    return whisper_inputs["input_features"], tokenized["input_ids"], tokenized["attention_mask"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", help="HF split")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    args = parser.parse_args()

    whisper_processor, whisper_model, llama_tokenizer, llama_model = load_processors_and_models()

    model = SpeechToLlamaModel(
        whisper_model=whisper_model,
        llama_model=llama_model
    ).to(DEVICE)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))

    model.eval()
    wer_metric = evaluate.load("wer")

    # Load dataset
    ds = load_asr_dataset(split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Evaluate WER
    all_preds = []
    all_refs = []
    prompt_text = PROMPT_TEXT if USE_PROMPT else ""

    for start in range(0, len(ds), args.batch_size):
        end = min(start + args.batch_size, len(ds))
        batch_ds = ds.select(range(start, end))
        audio_arrays = [ex["audio"]["array"] for ex in batch_ds]
        references = [ex["text"] for ex in batch_ds]

        input_features, input_ids, attention_mask = build_batch_inputs(
            whisper_processor,
            llama_tokenizer,
            audio_arrays,
            prompt_text,
        )

        input_features = input_features.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            encoder_outputs = model.whisper.encoder(input_features=input_features)
            speech_hidden = encoder_outputs.last_hidden_state

            # Match training downsampling
            speech_hidden = speech_hidden.transpose(1, 2)
            speech_hidden = F.avg_pool1d(
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

        preds = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(references)

    wer_value = wer_metric.compute(predictions=all_preds, references=all_refs)
    print(f"WER: {wer_value:.4f}")


if __name__ == "__main__":
    main()
