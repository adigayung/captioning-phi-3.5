import os
from transformers import AutoModelForImageClassification, AutoTokenizer

# Dapatkan path dari variabel lingkungan HF_HOME
hf_home = os.getenv('HF_HOME', './HuggingFace_HOME')
model_name = "microsoft/Phi-3.5-vision-instruct"

# Tentukan path model di dalam HF_HOME
model_dir = os.path.join(hf_home, model_name.replace("/", "_"))

# Cek apakah model sudah ada di direktori lokal
if os.path.exists(model_dir) and any(os.listdir(model_dir)):
    print("Model sudah tersedia")
else:
    print("Download model...")
    # Unduh model dan tokenizer ke HF_HOME
    model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=hf_home)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_home)
    print("Download model sukses")

# Muat model dan tokenizer dari HF_HOME
model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=hf_home)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_home)

print("Model dan tokenizer berhasil dimuat")
