
# Change to the data directory
cd data

# Download the Kaggle competition data
echo "Downloading competition data..."
kaggle competitions download bttai-ajl-2025

# Unzip the downloaded file
echo "Unzipping data..."
unzip -q bttai-ajl-2025.zip -d bttai-ajl-2025

rm bttai-ajl-2025.zip

echo "Download and extraction complete!"