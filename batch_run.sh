#!/bin/bash

# Define input directory
INPUT_DIR="partnet_selected"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found."
    exit 1
fi

# Iterate over all .png files in the directory
for img_path in "$INPUT_DIR"/*.png; do
    # Check if any .png files exist
    if [ ! -e "$img_path" ]; then
        echo "No .png files found in '$INPUT_DIR'."
        break
    fi

    # Extract filename without extension for session name
    filename=$(basename -- "$img_path")
    filename_no_ext="${filename%.*}"
    
    echo "========================================="
    echo "Processing: $filename"
    echo "========================================="
    
    # Run inference script
    # Using filename as session_name ensures outputs are saved in separate folders
    python run_inference.py \
        --image "$img_path" \
        --session_name "output_${filename_no_ext}" \
        --threshold 1200 \
        --seed 42 \
        --cfg 7.5
        
    if [ $? -eq 0 ]; then
        echo "Success: $filename processed."
    else
        echo "Error: Failed to process $filename."
    fi
    
    echo ""
done

echo "Batch processing complete."
