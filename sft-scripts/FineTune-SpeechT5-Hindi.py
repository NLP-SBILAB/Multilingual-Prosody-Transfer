from datasets import load_dataset, Audio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from speechbrain.pretrained import EncoderClassifier
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load the Hindi dataset from Common Voice
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train")
print(f"Total samples in dataset: {len(dataset)}")

# Cast the audio column to ensure correct sampling rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load the model and tokenizer
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

# Clean and normalize text data
def cleanup_text(batch):
    replacements = [
        ("।", ".")
    ]
    text = batch["sentence"]
    for src, dst in replacements:
        text = text.replace(src, dst)
    batch["normalized_text"] = text
    return batch

dataset = dataset.map(cleanup_text, remove_columns=["sentence"])

# Filter by speaker variability
speaker_counts = defaultdict(int)

for speaker_id in dataset["client_id"]:
    speaker_counts[speaker_id] += 1

def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

dataset = dataset.filter(select_speaker, input_columns=["client_id"])
print(f"Unique speakers selected: {len(set(dataset['client_id']))}")

# Load speaker recognition model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

# Process the dataset
def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    example["labels"] = example["labels"][0]
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
    return example

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Data collation
@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
        del batch["decoder_attention_mask"]
        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

# Initialize and configure model for training
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
model.config.use_cache = False
training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_common_voice_hindi",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

# Start training
trainer.train()
