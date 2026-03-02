# src/components/model_trainer.py
import os
import pandas as pd
from pathlib import Path
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Wav2Vec2FeatureExtractor, AutoModelForAudioClassification,
    TrainingArguments, Trainer
)
import torch


class UnifiedModelTrainer:
    def __init__(self, config, label_list):
        self.config = config
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # TEXT TRAINING
    # ======================
    def train_text_model(self, df):
        print("📝 Training DistilBERT on text (CPU-scaled)...")

        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name_text"])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name_text"],
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        ).to(self.device)

        # Encode text samples
        def encode(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=64  # smaller max length for CPU
            )
            enc["labels"] = [float(v) for v in batch["label_vec"]]
            return enc

        # Filter text modality
        dataset = Dataset.from_pandas(df[df["modality"] == "text"].reset_index(drop=True))
        dataset = dataset.map(encode, batched=False)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        output_path = self.output_dir / "distilbert"
        output_path.mkdir(parents=True, exist_ok=True)

        # ✅ CPU-optimized TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            per_device_train_batch_size=int(self.config.get("batch_size", 2)),
            num_train_epochs=int(self.config.get("max_epochs", 2)),
            learning_rate=float(self.config.get("lr", 2e-5)),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_strategy="epoch",        # ✅ Save after every epoch
            save_total_limit=None,        # ✅ Keep every checkpoint
            load_best_model_at_end=False,
            report_to="none",
            fp16=False,                   # ✅ Disable mixed precision
            dataloader_num_workers=0,     # ✅ Avoid multiprocessing overhead
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        # ✅ Auto-resume logic
        last_checkpoint = None
        if os.path.isdir(output_path):
            checkpoints = [
                d for d in os.listdir(output_path)
                if d.startswith("checkpoint-") and os.path.isdir(output_path / d)
            ]
            if checkpoints:
                last_checkpoint = str(output_path / sorted(checkpoints)[-1])
                print(f"🔁 Found checkpoint: {last_checkpoint}. Resuming training...")

        trainer.train(resume_from_checkpoint=last_checkpoint)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("✅ Text model training completed and saved!")

    # ======================
    # SPEECH TRAINING
    # ======================
    def train_speech_model(self, df):
        print("🎤 Training Wav2Vec2 on speech (CPU-scaled)...")

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config["model_name_speech"]
        )
        model = AutoModelForAudioClassification.from_pretrained(
            self.config["model_name_speech"],
            num_labels=self.num_labels
        ).to(self.device)

        # Placeholder encode for speech
        def encode(batch):
            return {
                "labels": [float(v) for v in batch.get("label_vec", [])]
            }

        dataset = Dataset.from_pandas(df[df["modality"] == "speech"].reset_index(drop=True))
        # dataset = dataset.map(encode)  # enable when audio processing ready

        output_path = self.output_dir / "wav2vec2"
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            per_device_train_batch_size=int(self.config.get("batch_size", 2)),
            num_train_epochs=int(self.config.get("max_epochs", 2)),
            learning_rate=float(self.config.get("lr", 2e-5)),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=None,
            load_best_model_at_end=False,
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        last_checkpoint = None
        if os.path.isdir(output_path):
            checkpoints = [
                d for d in os.listdir(output_path)
                if d.startswith("checkpoint-") and os.path.isdir(output_path / d)
            ]
            if checkpoints:
                last_checkpoint = str(output_path / sorted(checkpoints)[-1])
                print(f"🔁 Found checkpoint: {last_checkpoint}. Resuming training...")

        # trainer.train(resume_from_checkpoint=last_checkpoint)  # uncomment after audio input setup
        model.save_pretrained(output_path)
        feature_extractor.save_pretrained(output_path)
        print("✅ Speech model setup saved! (Training skipped for now)")

    # ======================
    # RUN METHOD
    # ======================
    def run(self, parquet_path=None):
        if parquet_path is None:
            parquet_path = self.config.get("transformed_path")
            if parquet_path is None:
                raise ValueError("No parquet_path supplied and 'transformed_path' missing in config.")

        print("🔄 Loading transformed dataset...")
        df = pd.read_parquet(parquet_path)
        print(f"Data shape: {df.shape}")

        # if "text" in df["modality"].values:
        #     self.train_text_model(df)

        if "speech" in df["modality"].values:
            self.train_speech_model(df)


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    transformed_path = cfg["trainer"].get("transformed_path")
    if transformed_path is None:
        raise KeyError("Please set trainer.transformed_path in config.yaml to the transformed parquet file path.")

    df = pd.read_parquet(transformed_path)
    unique_labels = sorted({l for sub in df["labels"] for l in sub})

    trainer = UnifiedModelTrainer(cfg["trainer"], unique_labels)
    trainer.run(transformed_path)
