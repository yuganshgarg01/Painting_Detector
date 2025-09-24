from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import transforms
from train_model import ArtModel, extract_pdf_text, transform
import pandas as pd

app = Flask(__name__)

# Load model and label map
dataset_csv = 'dataset.csv'
data = pd.read_csv(dataset_csv)
label_map = {label: idx for idx, label in enumerate(data['label'].unique())}
num_classes = len(label_map)
model = ArtModel(num_classes)
model.load_state_dict(torch.load('art_model.pth', map_location=torch.device('cpu')))
model.eval()

# Load PDF texts
pdf_texts = {
    'Gond': extract_pdf_text('pdfs/gond_art.pdf'),
    'Phad': extract_pdf_text('pdfs/phad_art.pdf'),
    # Add more schools as needed
}

def get_significance(school, pdf_texts):
    # Mock significance (replace with real BERT analysis)
    return (
        f"{school} art reflects cultural heritage. "
        f"{'Gond symbolizes tribal folklore and nature worship' if 'Gond' in school else 'Phad narrates Rajput epics, used by Bhopas in rituals'}."
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'analysis': 'No image uploaded!'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Model inference
    with torch.no_grad():
        logits = model(image_tensor)
        pred_idx = logits.argmax(1).item()
        school = list(label_map.keys())[list(label_map.values()).index(pred_idx)]
    
    # Mock feature extraction (replace with Grad-CAM for real features)
    features = "Vibrant dots, swirling motifs" if school == "Gond" else "Bold red/yellow lines, crowded scenes"
    is_indian = school in ['Gond', 'Pithora', 'Bhimbetka', 'Mandana', 'Phad', 'Miniature', 'Pichwai']
    is_mp = school in ['Gond', 'Pithora', 'Bhimbetka', 'Mandana']
    
    similarities = (
        f"Matches {school} examples in Indian Paintings Dataset and FolkTalent. "
        f"Unlike {'Bhimbetka’s earthy rock art' if is_mp else 'Gond’s dotted motifs'}. "
    )
    significance = get_significance(school, pdf_texts)
    
    analysis = (
        f"Wow, this is a stunning piece! Let’s dive in…\n"
        f"**Features**: {features}\n"
        f"**Is it Indian?**: {'Yes' if is_indian else 'No'}, "
        f"{'from Madhya Pradesh' if is_mp else 'from Rajasthan, not MP’s tribal styles'}.\n"
        f"**School**: {school}\n"
        f"**Similarities**: {similarities}\n"
        f"**Significance**: {significance}\n"
        f"Want to compare it to other {school} works or upload another?"
    )
    return jsonify({'analysis': analysis})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)