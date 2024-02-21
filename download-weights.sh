#!/bin/bash

file_url="https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
download_dir="weights"
mkdir -p "$download_dir"
filename="model.ckpt"

if [ -e "$download_dir/$filename" ]; then
    echo "File $filename already exists. Skipping download."
else
    echo "Downloading $filename..."
    wget -O "$download_dir/$filename" "$file_url"
    echo "Download complete."
fi
