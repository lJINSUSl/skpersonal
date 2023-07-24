from typing import Union
from fastapi import FastAPI, UploadFile, File
import torch
from starlette.requests import Request
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.requests import Request
from starlette.responses import HTMLResponse

from PIL import Image
import numpy as np
import io
import torch
from torchvision import datasets, transforms, models
import pandas as pd
from torchvision.io import read_image

transform = transforms.Compose([
    transforms.Resize((256,256)),

    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.ColorJitter(brightness=(0.5, 0.9),
    #                        contrast=(0.4, 0.8),
    #                        saturation=(0.7, 0.9),
    #                        hue=(-0.2, 0.2),
    #                       ),
    #transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])





device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

model.to(device)
model.load_state_dict(torch.load('model_2.pt'))


app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 불러오기 및 전처리
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    print(image.shape)
    print(image.size())
    print(image.dim())
    # 예측
    pred = model(image)
    _, result = torch.max(pred.data,1)

    return {"result": int(result)}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
        <body>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


from pyngrok import ngrok
import uvicorn
import threading

url = ngrok.connect(8000)  # FastAPI의 기본 포트는 8000입니다.
print('Public URL:', url)

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run).start()
