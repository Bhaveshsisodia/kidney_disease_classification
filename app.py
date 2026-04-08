


import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from kidney_disease_classifier.pipeline.prediction import PredictionPipeline
from kidney_disease_classifier.utils.common import decodeImage

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# ✅ GLOBAL VARIABLE
clApp = None

# ✅ CLIENT CLASS
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# ✅ LAZY LOADING FUNCTION
def get_model():
    global clApp
    if clApp is None:
        print("Loading model...")
        clApp = ClientApp()
        print("Model loaded")
    return clApp

# ✅ HOME ROUTE
@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")

# ✅ PREDICT ROUTE (FIXED)
@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    model = get_model()   # 👈 FIX

    image = request.json["image"]
    decodeImage(image, model.filename)

    result = model.classifier.predict()
    return jsonify(result)

@app.route("/train", methods=["GET","POST"])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



# ✅ MAIN (only for local)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

