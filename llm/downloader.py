from huggingface_hub import hf_hub_download

# model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
# model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
model_name = "andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_file = "TinyLlama-1.1B-Chat-v1.0-f16.gguf"
model_path = hf_hub_download(model_name, filename=model_file, local_dir="./")
