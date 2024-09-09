#!/bin/bash

# Cek apakah folder HuggingFace_HOME sudah ada
if [ -d "./HuggingFace_HOME" ]; then
    echo "Folder HuggingFace_HOME sudah ada."
else
    echo "Folder HuggingFace_HOME tidak ditemukan. Membuat folder..."
    mkdir ./HuggingFace_HOME
    echo "Folder HuggingFace_HOME berhasil dibuat."
fi

# Ambil path direktori saat ini
base_path=$(pwd)

# Set variabel lingkungan HF_HOME menggunakan base path
export HF_HOME="$base_path/HuggingFace_HOME"

echo "HF_HOME Set ke $HF_HOME"

# Unduh dan instal Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Tambahkan Miniconda ke PATH dan aktifkan
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash
source ~/.bashrc

# Buat lingkungan conda dan aktifkan
conda create -n myenv python=3.11 -y
conda activate myenv

# Cek apakah conda sudah terinstall
if ! command -v conda &> /dev/null; then
    echo "Conda tidak ditemukan. Pastikan Conda sudah terinstall."
    exit 1
fi

# Cek apakah conda-forge sudah ada dalam daftar channel
if conda config --show channels | grep -q "conda-forge"; then
    echo "Channel conda-forge sudah ada."
else
    echo "Menambahkan channel conda-forge..."
    conda config --add channels conda-forge
    echo "Channel conda-forge berhasil ditambahkan."
fi

# Install dependensi
pip install numpy==1.24.4
pip install Pillow==10.3.0
pip install Requests==2.31.0
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu117
pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.43.0
pip install accelerate==0.30.0
pip install flash_attn==2.5.8

# Jalankan skrip Python
python model-load.py
