from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO('yolov8n.pt')
names = model.names

@app.post('/predict')
async def busca_michis(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content).convert('RGB'))
    results = model(image)
    clase_yolo = [names[int(clase)] for clase in results[0].boxes.cls]
    return {"prediction": results}
