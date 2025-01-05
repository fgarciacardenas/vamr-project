#!/bin/bash

# === Configuration ===

# Destination directory where files will be downloaded and extracted
DEST_PATH="../data"

# URLs of the files to download
URLS=(
    "https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip"
    "https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip"
    "https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip"
)

# === Function Definitions ===

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# === Preliminary Checks ===

# Ensure wget is installed
if ! command_exists wget; then
    echo "Error: 'wget' is not installed. Please install it and retry."
    exit 1
fi

# Ensure unzip is installed
if ! command_exists unzip; then
    echo "Error: 'unzip' is not installed. Please install it and retry."
    exit 1
fi

# === Script Execution ===

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_PATH"

# Array to keep track of downloaded ZIP files
DOWNLOAD_ZIPS=()

# Loop through each URL and download the file
for URL in "${URLS[@]}"; do
    FILE_NAME="$(basename "$URL")"
    echo "Downloading $FILE_NAME..."
    
    wget -c "$URL" -P "$DEST_PATH"
    
    # Check if the download was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download $URL"
        exit 1
    fi
    
    # Add the downloaded ZIP file to the array
    DOWNLOAD_ZIPS+=("$DEST_PATH/$FILE_NAME")
done

echo "All files have been downloaded successfully to $DEST_PATH."

# Unzip each downloaded ZIP file and remove the original ZIP
for ZIP_FILE in "${DOWNLOAD_ZIPS[@]}"; do
    # Check if the ZIP file exists
    if [ ! -f "$ZIP_FILE" ]; then
        echo "Warning: $ZIP_FILE does not exist. Skipping."
        continue
    fi
    
    echo "Unzipping $(basename "$ZIP_FILE")..."
    
    # Unzip the file into the destination directory
    unzip -o "$ZIP_FILE" -d "$DEST_PATH"
    
    # Check if unzip was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to unzip $ZIP_FILE"
        exit 1
    fi
    
    # Remove the original ZIP file after successful extraction
    rm "$ZIP_FILE"
    
    echo "Successfully extracted and removed $ZIP_FILE."
done

echo "All files have been unzipped and original ZIP files have been removed."
