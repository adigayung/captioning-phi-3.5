import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import requests
from io import BytesIO
import os

def process_image(image, model, processor, filename, current_index, total_images, output_dir):
    # Mengatur pesan untuk prompt per gambar
    placeholder = "<|image_1|>\n"
    messages = [{"role": "user", "content": placeholder + "Describe this image in detail, including the main subject, actions, and setting."}]
    
    # Membuat prompt berdasarkan template chat
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Proses input dan kirim ke GPU
    inputs = processor(prompt, images=[image], return_tensors="pt").to("cuda:0")
    
    # Argumen generasi
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50
    }

    # Menghasilkan deskripsi
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Hapus token input dari hasil generate
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Menghapus frasa "The image features"
    response = response.replace("The image features", "").strip()

    # Cetak deskripsi gambar
    print(f"\n[{current_index}/{total_images}] Describing the image: {filename}")
    print(response)

    # Simpan hasil deskripsi ke file dengan nama yang sama seperti gambar di dalam folder yang sama
    text_filename = os.path.splitext(filename)[0] + ".txt"
    text_filepath = os.path.join(output_dir, text_filename)
    with open(text_filepath, "w") as f:
        f.write(response)

def main():
    # Inisialisasi parser argumen
    parser = argparse.ArgumentParser(description="Generate image descriptions using a pre-trained model.")
    parser.add_argument('--image-dir', type=str, help='Path to a directory containing images.')
    parser.add_argument('--url-image', type=str, help='URL of an image to describe.')
    parser.add_argument('--image-file', type=str, help='Path to a specific image file.')

    # Parse argumen yang diberikan
    args = parser.parse_args()

    # Jika tidak ada argumen yang diberikan, print help dan keluar tanpa load model
    if not (args.image_dir or args.url_image or args.image_file):
        parser.print_help()
        return

    # Bersihkan cache GPU
    torch.cuda.empty_cache()

    # ID model yang digunakan
    model_id = "microsoft/Phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='eager'
    )

    # Prosesor
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=16
    )

    if args.image_dir:
        # Dapatkan total jumlah gambar
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        # Proses semua gambar dalam direktori satu per satu
        for idx, filename in enumerate(image_files):
            img_path = os.path.join(args.image_dir, filename)
            img = Image.open(img_path)
            process_image(img, model, processor, filename, idx+1, total_images, args.image_dir)
    elif args.url_image:
        # Proses gambar dari URL
        response = requests.get(args.url_image)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            process_image(img, model, processor, "image-from-url.jpg", 1, 1, ".")
        else:
            print(f"Failed to download image, status code: {response.status_code}")
    elif args.image_file:
        # Proses file gambar tertentu
        img = Image.open(args.image_file)
        filename = os.path.basename(args.image_file)
        output_dir = os.path.dirname(args.image_file)  # Simpan di folder yang sama dengan gambar
        process_image(img, model, processor, filename, 1, 1, output_dir)

if __name__ == "__main__":
    main()
