import torch
import cv2
import numpy as np
from core.s_img_data_model import CNN
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def predict_image(img_path, model_path=None, img_size=128):
    if model_path is None:
        model_path = PROJECT_ROOT / "models/cnn_model.pth"

    cnn = CNN(img_size=img_size)
    cnn.load_state_dict(torch.load(model_path))
    cnn.eval()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # channel
    img = np.expand_dims(img, axis=0)  # batch
    img_tensor = torch.tensor(img)

    with torch.no_grad():
        output = cnn(img_tensor)
        prediction = output.item() > 0.5

    return "not_spores" if prediction else "spores"
