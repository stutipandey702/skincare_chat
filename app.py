import os
import sys
import re
import json
import threading

# Set cache dirs BEFORE importing transformers
_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HOME"] = _cache_dir
os.environ["TRANSFORMERS_CACHE"] = _cache_dir
os.environ["HF_HUB_CACHE"] = _cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = _cache_dir
os.environ["HF_DATASETS_CACHE"] = _cache_dir

cache_dir = _cache_dir
os.makedirs(cache_dir, exist_ok=True)

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import torch

app = Flask(__name__)

# ── Device selection: prefer MPS (Apple Silicon GPU) over CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE  = torch.float16        # float16 runs well on MPS
    print("Using Apple Silicon GPU (MPS) with float16")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE  = torch.float16
    print("Using CUDA GPU with float16")
else:
    DEVICE = torch.device("cpu")
    DTYPE  = torch.float32        # float16 can be unstable on CPU
    print("Falling back to CPU — inference will be slow")


def load_model():
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_model_id = "stutipandey/llama_skinchat_lora"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_fast=True,          # fast tokenizer is fine here and faster
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        torch_dtype=DTYPE,      # was float32 — now float16 on GPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    model = model.to(DEVICE)    # move to MPS/CUDA/CPU
    model.eval()

    print(f"Model loaded on {DEVICE}")
    return tokenizer, model


print("Loading model...")
try:
    tokenizer, model = load_model()
    model_type = "TinyLlama + LoRA"
except Exception as e:
    print(f"Model load failed: {e}")
    sys.exit(1)


def build_prompt(user_text: str) -> str:
    return (
        f"<|system|>\nYou are a helpful skincare assistant.</s>\n"
        f"<|user|>\n{user_text}</s>\n"
        f"<|assistant|>\n"
    )


def clean_response(raw: str) -> str:
    """Strip prompt prefix and trim to the last complete sentence."""
    if "<|assistant|>" in raw:
        raw = raw.split("<|assistant|>")[-1].strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    if sentences and not raw.rstrip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    return " ".join(sentences).strip()


# ── Non-streaming endpoint (kept for compatibility) ───────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    formatted = build_prompt(prompt)
    inputs    = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)            # inputs must be on same device as model

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    raw      = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(raw)
    return jsonify({"response": response, "model_used": model_type})


# ── Streaming endpoint (new — much better perceived speed)
@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    """
    Returns a text/event-stream (SSE) response so the frontend can render
    tokens as they arrive rather than waiting for the full generation.
    """
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    formatted = build_prompt(prompt)
    inputs    = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # TextIteratorStreamer yields tokens as they are produced
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,           # don't echo the input back
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    # Run generation in a background thread so we can stream from main thread
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def token_stream():
        for token in streamer:
            # SSE format: "data: <payload>\n\n"
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(token_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
        },
    )


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": model_type,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
    })


if __name__ == "__main__":
    print(f"Server starting — model: {model_type} | device: {DEVICE}")
    app.run(host="0.0.0.0", port=7860, debug=False)