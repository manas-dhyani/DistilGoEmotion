# src/data_ingestion.py
import os
import sys
from pathlib import Path
import datasets
#import torchaudio
import pandas as pd

class UnifiedDataIngestion:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)

    def load_goemotions(self):
        """
            Loads GoEmotions dataset (raw config with one-hot columns).
            Returns: list of dicts with fields {id, modality, text, audio, labels}
      """
    
        dataset = datasets.load_dataset("go_emotions", "raw")

    # all emotion columns (multi-hot format)
        emotion_cols = [c for c in dataset["train"].column_names 
                        if c not in ["text", "id", "author", "subreddit", "link_id", 
                                     "parent_id", "created_utc", "rater_id", "example_very_unclear"]]

        print(f"Detected {len(emotion_cols)} emotion columns: {emotion_cols[:5]}...")

        all_data = []
        for split in dataset.keys():  # e.g., only "train" exists in raw
            for idx, row in enumerate(dataset[split]):
            # find all emotion labels that are 1
                labels = [col for col in emotion_cols if row[col] == 1]

                all_data.append({
                "id": f"goemotions-{split}-{idx}",
                "modality": "text",
                "text": row["text"],
                "audio": None,
                "labels": labels
            })
        return all_data
    

    def load_iemocap_hf(self):
        dataset = datasets.load_dataset("AbstractTTS/IEMOCAP")  # no casting, no torchaudio needed

        print("Splits:", dataset.keys())
        print("Columns in train:", dataset["train"].column_names)

        # Use pandas view of the arrow dataset to avoid audio decoding
        # (this will load metadata + paths as python objects)
        split = list(dataset.keys())[0]  # usually 'train' for this HF copy
        pdf = dataset[split].to_pandas()

        # Safe sample inspection
        sample = pdf.iloc[0]
        print("Sample transcription:", sample.get("transcription"))
        # prefer 'file' column for path if present; otherwise check sample['audio']
        audio_path = sample.get("file") or (sample.get("audio") and sample["audio"].get("path"))
        print("Sample audio path:", audio_path)
        print("Sample major_emotion:", sample.get("major_emotion"))

        all_data = []
        for split in dataset.keys():
            pdf = dataset[split].to_pandas()
            for idx, row in pdf.iterrows():
                audio_path = row.get("file") or (row.get("audio") and row["audio"].get("path"))
                all_data.append({
                "id": f"iemocap-{split}-{idx}",
                "modality": "speech",
                "text": row.get("transcription", "") or row.get("text", ""),
                "audio": audio_path,
                "labels": [row.get("major_emotion")] if row.get("major_emotion") is not None else []
            })
        return all_data

    

    def _extract_annotation(self, wav_path: Path, wav_id: str):
        """Find transcript + emotion for given wav file (simplified)."""
        session_dir = wav_path.parents[2]  # e.g., Session1/dialog/...
        transcript, emotion = "", "neutral"

        emo_file = session_dir / "dialog/EmoEvaluation" / f"{wav_id}.txt"
        if emo_file.exists():
            with open(emo_file) as f:
                for line in f:
                    if wav_id in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            transcript = parts[1]
                            emotion = parts[2]
        return transcript, emotion

    def run(self):
        print("Loading GoEmotions...")
        goemotions_data = self.load_goemotions()

        print("Loading IEMOCAP...")
        iemocap_data = self.load_iemocap_hf()

        all_data = goemotions_data + iemocap_data
        df = pd.DataFrame(all_data)

        output_path = Path(self.config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        print(f"✅ Unified dataset saved at {output_path}")
        return str(output_path)  # return path, not dataframe

        # src/data_ingestion.py

if __name__ == "__main__":
    import yaml

    # Load config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Instantiate and run ingestion
    ingestion = UnifiedDataIngestion(config["ingestion"])
    output_path = ingestion.run()

    print(f"✅ Finished! Data saved to {output_path}")
    