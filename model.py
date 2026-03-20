import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechToLlamaModel(nn.Module):
    def __init__(self, whisper_model, llama_model, freeze_whisper=True, freeze_llama=True):
        super().__init__()
        self.whisper = whisper_model
        self.llama = llama_model
        
        d_whisper = self.whisper.config.d_model
        d_llama = self.llama.config.hidden_size

        # Projector W  (d_whisper = 512, d_llama = 2048)
        self.projector = nn.Sequential(
            nn.Linear(d_whisper, d_llama),
            # GELU to add non-linearity (better approximation)
            nn.GELU(),
            nn.Linear(d_llama, d_llama),
        )


    def forward(self, input_features, input_ids, attention_mask):
        # Whisper encoder
        encoder_outputs = self.whisper.encoder(input_features=input_features)
        speech_hidden = encoder_outputs.last_hidden_state

        # Downsampling to shorten sequence length

        speech_hidden = speech_hidden.transpose(1, 2)
        speech_hidden = F.avg_pool1d(
            speech_hidden,
            kernel_size=4,
            stride=4,
        )
        speech_hidden = speech_hidden.transpose(1, 2)

        # Projector W)
        speech_embeds = self.projector(speech_hidden)

        # Text embeddings
        llama_embed = self.llama.get_input_embeddings()
        text_embeds = llama_embed(input_ids)

        # Match dtype (Llama was in bf16)
        target_dtype = text_embeds.dtype
        if speech_embeds.dtype != target_dtype:
            speech_embeds = speech_embeds.to(target_dtype)

        # Concatenate speech and text embeddings
        inputs_embeds = torch.cat([speech_embeds, text_embeds], dim=1)

        batch_size, audio_len, _ = speech_embeds.shape

        speech_mask = torch.ones(
            batch_size,
            audio_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([speech_mask, attention_mask], dim=1)

        prefix_labels = torch.full(
            (batch_size, audio_len),
            -100,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return outputs