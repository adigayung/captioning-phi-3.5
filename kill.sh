#!/bin/bash

# Cek proses yang menggunakan GPU
echo "Menemukan proses yang menggunakan GPU..."

# Gunakan nvidia-smi untuk menemukan proses GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    if [ -n "$pid" ]; then
        echo "Menghentikan proses dengan PID: $pid"
        kill -9 $pid
    fi
done

echo "Semua proses GPU dihentikan."
