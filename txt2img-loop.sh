#!/bin/bash

while getopts ":c:p:o:" opt; do
  case $opt in
    c) ckpt_directory="$OPTARG" ;;
    p) prompt="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

python_script="txt2img.py"

if [ -z "$ckpt_directory" ]; then
  echo "Error: Please provide the directory path as an argument."
  exit 1
fi

if [ ! -d "$ckpt_directory" ]; then
  echo "Error: Directory '$ckpt_directory' does not exist."
  exit 1
fi

for ckpt in "$ckpt_directory"/*; do
  if [ -f "$ckpt" ]; then
    python "$python_script" -o "$out_dir" -c "$ckpt" -p "$prompt" -v 8 -b 4 -d 1
  fi
done

echo "Looped inferencing complete!"
