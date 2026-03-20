import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from collator import SpeechCollator
from config import DEVICE, EVAL_BATCH_SIZE, LR, NUM_EPOCHS, TRAIN_BATCH_SIZE
from dataset import load_asr_dataset
from model import SpeechToLlamaModel
from utils import load_processors_and_models


def setup_distributed():
    if not dist.is_available():
        return False, 0
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, local_rank


def main():
    is_distributed, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else DEVICE)
    whisper_processor, whisper_model, llama_tokenizer, llama_model = load_processors_and_models()

    # For testing tried with 1 GPU
    # train_ds = load_asr_dataset(split="train.100", max_samples=200)
    # val_ds = load_asr_dataset(split="validation", max_samples=50)

    # using all 8 GPUs
    train_ds = load_asr_dataset(split="train.100", max_samples=5000)
    val_ds   = load_asr_dataset(split="validation", max_samples=500)

    collator = SpeechCollator(whisper_processor, llama_tokenizer)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
    )

    model = SpeechToLlamaModel(
        whisper_model=whisper_model,
        llama_model=llama_model,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    writer = None
    if not is_distributed or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir="runs/speech_llama")
        writer.add_text("config/model", f"whisper={type(whisper_model).__name__}, llama={type(llama_model).__name__}")

    # Train loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model(
                input_features=batch["input_features"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0 and (not is_distributed or dist.get_rank() == 0):
                print(f"epoch={epoch} step={step} train_loss={loss.item():.4f}")
                if writer is not None:
                    global_step = epoch * len(train_loader) + step
                    writer.add_scalar("train/loss", loss.item(), global_step)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(
                    input_features=batch["input_features"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                val_losses.append(outputs.loss.item())

        avg_val = sum(val_losses) / max(len(val_losses), 1)
        if not is_distributed or dist.get_rank() == 0:
            print(f"epoch={epoch} val_loss={avg_val:.4f}")
            if writer is not None:
                writer.add_scalar("eval/loss", avg_val, epoch)
            state = model.module.state_dict() if is_distributed else model.state_dict()
            torch.save(state, f"checkpoints/model_epoch_{epoch}.pt")

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
