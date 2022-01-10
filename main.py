import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image
from distilfnd import DistilFND
import torch

import time

app = Flask(__name__)
CORS(app)

distil = DistilFND(6)

image_name = ""

@app.route("/", methods=['GET'])
def initApp():
    distil.load_model()
    time.sleep(3)
    result = "start"
    return jsonify({"result": result})

@app.route("/predict", methods=['POST'])
def predict():
    text = request.json['text']
    image = Image.open(f"{image_name}")

    title_input_ids, title_attention_mask, comment_input_ids, comment_attention_mask = distil.tokenize(text, "")
    image_tensor = distil.process_image(image)

    distil.eval()
    prediction_probs = distil(
        title_input_ids=title_input_ids,
        title_attention_mask=title_attention_mask,
        image=image_tensor,
        cm_input_ids=comment_input_ids,
        cm_attention_mask=comment_attention_mask
    )
    _, prediction = torch.max(prediction_probs, dim=1)

    if prediction.item() == 0:
        return jsonify({"result": "Predicted : True Content"})

    return jsonify({"result": "Predicted : Fake Content"})

@app.route("/download", methods=['POST'])
def download():
    result = request.files['photo']
    result.save(os.path.join(result.filename))
    global image_name
    image_name = result.filename
    print(result)
    return jsonify({"result": "pic"})

if __name__ == "__main__":
    app.run(ssl_context='adhoc', debug=True)
