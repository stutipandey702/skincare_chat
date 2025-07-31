from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stutipandey/llama_skinchat_lora",
    local_dir="llama_skinchat_lora",
    local_dir_use_symlinks=False
)

