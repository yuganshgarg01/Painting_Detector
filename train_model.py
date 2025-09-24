import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import numpy as np

# Extract text from PDFs for significance
def extract_pdf_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
    except:
        return ""

pdf_texts = {
    'Gond': extract_pdf_text('pdfs/gond_art.pdf'),
    'Phad': extract_pdf_text('pdfs/phad_art.pdf'),
    # Add more schools as needed
}

# Create dataset CSV
data_dir = 'dataset/'
kaggle_dir = 'data/kaggle/indian-paintings-dataset/'
image_paths = []
labels = []
for region in ['Madhya_Pradesh', 'Rajasthan']:
    for school in os.listdir(f'{data_dir}/{region}'):
        for img in os.listdir(f'{data_dir}/{region}/{school}'):
            if img.endswith(('.jpg', '.png')):
                image_paths.append(f'{data_dir}/{region}/{school}/{img}')
                labels.append(school)
for img in os.listdir(kaggle_dir):
    for school in ['Gond', 'Pithora', 'Phad', 'Miniature']:
        if school.lower() in img.lower():  # Basic filter; refine as needed
            image_paths.append(f'{kaggle_dir}/{img}')
            labels.append(school)
df = pd.DataFrame({'path': image_paths, 'label': labels})
df.to_csv('dataset.csv', index=False)

# Define dataset
class ArtDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.data['label'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return None, None  # Skip invalid images
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_map[label]
        return image, label_idx

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define model
class ArtModel(nn.Module):
    def __init__(self, num_classes):
        super(ArtModel, self).__init__()
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, images, text=None):
        img_features = self.cnn(images)
        if text:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text_features = self.text_model(**inputs).last_hidden_state[:, 0, :]
            return img_features, text_features
        return img_features

# Training setup
dataset = ArtDataset('dataset.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
num_classes = len(dataset.label_map)
model = ArtModel(num_classes).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        if images is None:  # Skip invalid batches
            continue
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

# Save model
torch.save(model.state_dict(), 'art_model.pth')
print("Model trained and saved as art_model.pth")