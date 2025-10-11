# src/components/data_transformation.py
import pandas as pd
import json
import numpy as np
from pathlib import Path

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_path: str):
        print(f"🔄 Loading unified dataset from {input_path}...")
        df = pd.read_parquet(input_path)

        # Collect all unique labels (strings)
        all_labels = set()
        for labels in df["labels"]:
            if labels is not None and len(labels) > 0:  # skip empty
                all_labels.update(labels)

        all_labels = sorted(list(all_labels))
        label2id = {label: i for i, label in enumerate(all_labels)}
        id2label = {i: label for label, i in label2id.items()}

        print(f"✅ Found {len(all_labels)} unique labels: {all_labels[:10]}...")

        # Convert string labels to integer indices
        df["label_ids"] = df["labels"].apply(
            lambda x: [label2id[label] for label in x if label in label2id]
        )

        # 🔑 Convert to multi-hot vectors for BCEWithLogitsLoss
        def to_multihot(label_ids, num_labels):
            vec = np.zeros(num_labels, dtype=int)
            for lid in label_ids:
                vec[lid] = 1
            return vec.tolist()

        df["label_vec"] = df["label_ids"].apply(lambda ids: to_multihot(ids, len(all_labels)))

        # Save transformed parquet (with multi-hot vectors)
        transformed_path = self.output_dir / "transformed_dataset.parquet"
        df.to_parquet(transformed_path, index=False)

        # Save label mapping
        labelmap_path = self.output_dir / "labels.json"
        with open(labelmap_path, "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

        print(f"✅ Transformed dataset saved at {transformed_path}")
        print(f"✅ Label mapping saved at {labelmap_path}")

        return str(transformed_path), str(labelmap_path)


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    transformer = DataTransformation(config["transformation"])
    transformer.run(config["ingestion"]["output_path"])
