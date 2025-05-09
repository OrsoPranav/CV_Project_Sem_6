# app.py
import io
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from torch.hub import load

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load pretrained DeepLabV3 model for Cityscapes
model = load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Cityscapes class colors (RGB)
CITYSCAPES_COLORS = [
    (128, 64, 128),   # Road
    (244, 35, 232),   # Sidewalk
    (70, 70, 70),     # Building
    (102, 102, 156),  # Wall
    (190, 153, 153),  # Fence
    (153, 153, 153),  # Pole
    (250, 170, 30),   # Traffic Light
    (220, 220, 0),    # Traffic Sign
    (107, 142, 35),   # Vegetation
    (152, 251, 152),  # Terrain
    (70, 130, 180),    # Sky
    (220, 20, 60),     # Person
    (255, 0, 0),       # Rider
    (0, 0, 142),       # Car
    (0, 0, 70),        # Truck
    (0, 60, 100),      # Bus
    (0, 80, 100),      # Train
    (0, 0, 230),       # Motorcycle
    (119, 11, 32)      # Bicycle
]

def preprocess(image):
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def decode_segmap(mask):
    rgb = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.uint8)
    for class_idx, color in enumerate(CITYSCAPES_COLORS):
        rgb[mask == class_idx] = torch.tensor(color, dtype=torch.uint8)
    return Image.fromarray(rgb.numpy())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            # Save original image
            img = Image.open(file.stream).convert('RGB')
            img_path = f"{app.config['UPLOAD_FOLDER']}/original.jpg"
            img.save(img_path)
            
            # Process image
            input_tensor = preprocess(img)
            with torch.no_grad():
                output = model(input_tensor)['out'][0]
            mask = output.argmax(0).cpu()
            
            # Create and save segmentation mask
            seg_img = decode_segmap(mask)
            seg_path = f"{app.config['UPLOAD_FOLDER']}/segmentation.jpg"
            seg_img.save(seg_path)
            
            return render_template('index.html', result=True)
    
    return render_template('index.html', result=False)

@app.route('/uploads/<filename>')
def send_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)


# templates/index.html (embedded using Flask's templating system)
