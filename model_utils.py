# model_utils.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import librosa

# Optionally limit threads on CPU (uncomment if your laptop feels overloaded)
# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# torch.set_num_threads(2)

# -------------------------
# Text emotion (DistilBERT-like)
# -------------------------
def load_text_model(model_name_or_path, device="cpu"):
    """
    Loads a text classification model (DistilBERT fine-tuned on GoEmotions).
    Returns (model, tokenizer, label_list). Model is moved to `device`.
    """
    model_name_or_path = "models/distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)#path updation
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)#path updation
    model.to(device)
    model.eval()

    # Try to obtain label list from config; else fallback to numeric labels
    label_list = None
    config = model.config
    if hasattr(config, "id2label") and config.id2label: # path updation
        id2label = config.id2label
        # ensure order by integer keys
        label_list = [id2label[i] for i in sorted(map(int, id2label.keys()))]
    return model, tokenizer, label_list

def predict_text_emotion(text, model, tokenizer, device="cpu"):
    """
    Returns (label, confidence, probs).
    Assumes model is already on the device (load_text_model moves it).
    """
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    # resolve label name
    label = None
    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = model.config.id2label # resolve path updation
        # id2label keys may be strings; coerce
        label = id2label[str(pred_id)] if str(pred_id) in id2label else id2label.get(pred_id, str(pred_id))
    else:
        label = str(pred_id)

    confidence = float(probs[pred_id])
    return label, confidence, probs

# -------------------------
# Speech emotion (Wav2Vec2)
# -------------------------
def load_speech_emotion_model(model_name_or_path, device="cpu"):
    """
    Loads feature extractor and Wav2Vec2 classification model and moves model to device.
    Returns (model, feature_extractor, label_list).
    """
    model_name_or_path = "models/wav2vec2-finetuned-final" #path updation
    feat = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path) #path updation
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path) #path updation
    model.to(device)
    model.eval()

    label_list = None
    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = model.config.id2label
        label_list = [id2label[i] for i in sorted(map(int, id2label.keys()))]
    return model, feat, label_list

def predict_speech_emotion(wav_path, model, feature_extractor, device="cpu"):
    """
    Predict emotion from audio file path.
    Assumes model is already on `device`.
    Returns (label, confidence, probs)
    """
    speech, sr = librosa.load(wav_path, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = model.config.id2label # resolve path updation
        label = id2label[str(pred_id)] if str(pred_id) in id2label else id2label.get(pred_id, str(pred_id))
    else:
        label = str(pred_id)

    confidence = float(probs[pred_id])
    return label, confidence, probs

# -------------------------
# ASR helpers
# -------------------------
def load_asr_pipeline(asr_model_name="facebook/wav2vec2-large-960h", device=-1):
    """
    Returns an HF pipeline for ASR. For CPU use device=-1.
    """
    asr = pipeline("automatic-speech-recognition", model=asr_model_name, chunk_length_s=10, device=device)
    return asr

def transcribe_audio(wav_path, asr_pipeline):
    result = asr_pipeline(wav_path)
    text = result.get("text", result) if isinstance(result, dict) else result
    return text

# -------------------------
# Local LLM helpers (CPU-friendly)
# -------------------------
def load_local_llm(model_name_or_path, model_type="text2text", device="cpu"):
    """
    Load a local LLM for CPU inference.
    model_type: "text2text" (e.g., flan-t5-small) or "causal" (e.g., distilgpt2)
    This returns (pipeline_fn, tokenizer, model) where pipeline_fn is a HF pipeline object.
    The returned pipeline uses device=-1 for CPU.
    """
    # For CPU set device=-1 in pipeline
    if model_type == "text2text":
        model_name_or_path = "models/google_flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model.to(device)
        pipeline_obj = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)
        pipeline_obj = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return pipeline_obj, tokenizer, model

def generate_from_llm(pipeline_obj, prompt, max_new_tokens=80, do_sample=False, temperature=0.7):
    """
    Generate text from the pipeline (CPU-friendly defaults).
    Returns the generated text string.
    """
    out = pipeline_obj(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    # pipelines return lists
    if isinstance(out, list) and len(out) > 0:
        # common key names: 'generated_text' or 'text'
        first = out[0]
        return first.get("generated_text") or first.get("text") or str(first)
    # fallback
    return str(out)

# -------------------------
# End of file
# -------------------------
