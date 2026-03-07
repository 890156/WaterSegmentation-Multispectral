
from flask import Flask, render_template, request
import torch
import numpy as np
from PIL import Image
import rasterio
import cv2

from model import load_model

app = Flask(__name__)

model = load_model()


def preprocess_image(path):

    with rasterio.open(path) as src:
        image = src.read()

    image = np.transpose(image, (1,2,0))
    image = image.astype(np.float32)

    image = torch.tensor(image).permute(2,0,1).unsqueeze(0)

    return image


def predict(image_tensor):

    with torch.no_grad():

        output = model(image_tensor)

        prob = torch.sigmoid(output)

        mask = (prob > 0.5).float()

        mask = mask.squeeze().cpu().numpy()

    return mask


@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]

        path = "static/input.tif"
        file.save(path)

        img = preprocess_image(path)

        mask = predict(img)

        mask = (mask * 255).astype(np.uint8)

        cv2.imwrite("static/output.png", mask)

        return render_template(
            "index.html",
            input_image="static/input.tif",
            output_image="static/output.png"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)