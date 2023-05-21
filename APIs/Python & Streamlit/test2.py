from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model('img.jpg', save = True)

clase_yolo = results[0].boxes.cls

names = results[0].names

clase_yolo = [names[int(clase)] for clase in clase_yolo]

print(clase_yolo)