# local_llm.py
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed
)

# ---------- Configuration ----------
# Choose one: "flan-t5-small" or "distilgpt2"
# FLAN (better instruction following, seq2seq)
FLAN = "google/flan-t5-small"
# DISTILGPT2 (smallest causal LM)
DISTIL = "distilgpt2"

# Choose preferred model here:
USE_MODEL = FLAN   # or DISTIL

# Where to save the model locally after first download (optional)
LOCAL_DIR = "./models/" + USE_MODEL.replace("/", "_")

# Speed-tuning: limit threads (helps on CPU)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
torch.set_num_threads(2)

# Deterministic seed for reproducibility (optional)
set_seed(42)

# ---------- Load model & tokenizer ----------
if USE_MODEL == FLAN:
    print(f"Loading seq2seq model {FLAN} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(FLAN)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN, device_map=None).to("cpu")
    # pipeline for text2text
    gen_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
else:
    print(f"Loading causal LM {DISTIL} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(DISTIL)
    model = AutoModelForCausalLM.from_pretrained(DISTIL, device_map=None).to("cpu")
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

print("Model loaded successfully. (If this is the first run, it downloaded model files)")

# Optionally save locally to avoid re-downloads
os.makedirs(LOCAL_DIR, exist_ok=True)
model.save_pretrained(LOCAL_DIR)
tokenizer.save_pretrained(LOCAL_DIR)
print("Saved model + tokenizer to", LOCAL_DIR)

# ---------- Helper generate function ----------
def generate(prompt, max_new_tokens=120, do_sample=False, temperature=0.7):
    """
    Generate text using the chosen pipeline.
    Use greedy generation (do_sample=False) for deterministic, faster results on CPU.
    """
    # For flan-t5 (text2text) the pipeline expects an input string; for distilgpt2 it's the same.
    outputs = gen_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    # pipeline returns list of dicts: [{'generated_text': '...'}]
    return outputs[0]["generated_text"]

# ---------- Quick interactive demo ----------
if __name__ == "__main__":
    print("\n=== Local LLM demo ===")
    print("Model:", USE_MODEL)
    print("Type a prompt (or press Enter to use a sample).")
    p = input("Prompt> ").strip()
    if not p:
        if USE_MODEL == FLAN:
            p = "You are a supportive counselor. A user says: \"I feel anxious about an upcoming interview.\" Give 2 short coping tips and a supportive sentence."
        else:
            p = "Write a short supportive message for someone anxious about an interview."

    print("\nGenerating (this may take a few seconds on CPU)...\n")
    out = generate(p, max_new_tokens=80, do_sample=False)
    print("=== Output ===\n")
    print(out)
