
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("models/damagedcarmodel.h5")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = read_file_as_image(await file.read())
    img_resized = tf.image.resize(img, (256,256))
    img_batch = np.expand_dims(img_resized/255, 0)
    
    predictions = MODEL.predict(img_batch)

    yhat = predictions[0][0]

    if yhat>=0.5:
        return 'Not Damaged'
    else:
        return 'Damaged'

    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)