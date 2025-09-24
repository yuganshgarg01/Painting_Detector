#!/bin/bash

# Create project directories
mkdir -p dataset/Madhya_Pradesh/{Gond,Pithora,Bhimbetka,Mandana}
mkdir -p dataset/Rajasthan/{Phad,Miniature,Pichwai,Mandana}
mkdir -p data/kaggle
mkdir -p pdfs

# Download Kaggle dataset (ensure kaggle.json is in ~/.kaggle/)
pip install kaggle
kaggle datasets download -d ajg117/indian-paintings-dataset -p data/kaggle --unzip
kaggle datasets download -d folk-talent/indian-folk-paintings -p data/kaggle --unzip  # Note: Hypothetical dataset; replace with actual if needed

# Placeholder for downloading PDFs (e.g., Gond, Phad research papers)
# Replace URLs with actual sources or manually place PDFs in pdfs/
curl -o pdfs/gond_art.pdf https://www.researchgate.net/publication/326479931_An_account_of_dots_and_lines-_The_Gond_Tribal_Art_of_Madhya_Pradesh
curl -o pdfs/phad_art.pdf https://www.exoticindiaart.com/book/details/tribal-painting-and-sculptures-had476/

echo "Setup complete! Place local images in dataset/ and PDFs in pdfs/"