
import pickle
import base64
import numpy as np
from io import BytesIO
from skimage import io
from flask import Flask, request, jsonify
from skimage.transform import resize

app = Flask(__name__)

def transform_base64(image_string):
    return np.array(resize(io.imread(BytesIO(base64.b64decode(image_string))), (15,15))).flatten()

def predict(image_string):
    model = pickle.load(open(r"./Model_File/finalized_model.sav", "rb"))
    return model.predict([transform_base64(image_string)])[0]

@app.route('/predict', methods=['POST'])
def predict_route():
    return jsonify({'prediction': predict(request.json['image'])})

app.run(host='localhost', port=5000, debug=True)