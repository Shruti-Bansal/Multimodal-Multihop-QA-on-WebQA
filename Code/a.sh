#!/bin/bash

# Specify the path to your data folder
data_folder="/home/ubuntu/Project/11-777-Project-aa/Data/Data"

# Change to the data folder
if [ ! -d "$data_folder" ]; then
  echo "Data folder does not exist."
  exit 1
fi

# Loop through each .7z file in the data folder
for file in "$data_folder"/*.7z; do
  if [ -e "$file" ]; then
    # Decompress the file
    7z x "$file" -o"$data_folder"

    # Delete the compressed file
    rm "$file"
  fi
done