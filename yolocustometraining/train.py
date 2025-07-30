from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data = 'datasets\data.yaml', epochs=30, imgsz=640, batch=8)
model.export(format='onnx')