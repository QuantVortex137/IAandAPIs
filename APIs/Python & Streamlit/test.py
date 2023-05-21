import requests

response = requests.get('http://localhost:8000/')
url = 'http://localhost:8000/predict'

image_path = 'image.jpg'

with open(image_path, 'rb') as f:
    image = f.read()

print(response.json())