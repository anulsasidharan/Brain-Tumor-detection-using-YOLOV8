from ultralytics import YOLO
from PIL import Image

model = YOLO(r'runs\detect\train\weights\best.pt')
model(r"C:\Users\Euron\Downloads\brain-tumor\images\val\val_1 (58).jpg", conf=0.007, save=True, show=True)