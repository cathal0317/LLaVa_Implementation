from datasets import Audio, load_dataset

from config import SAMPLE_RATE


def load_asr_dataset(split="train.100", max_samples=None):
    ds = load_dataset("librispeech_asr", "clean", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    def format_example(example):
        return {
            "audio": example["audio"],
            "text": example["text"],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds