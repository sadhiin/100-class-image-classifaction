import os
import cv2
import base64
from flask_cors import CORS, cross_origin
from LCIC.prediction_service import PredictionService
from LCIC.utils.common import decodeImage
from flask import Flask, render_template, request, flash, jsonify

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
app = Flask(__name__)
app.secret_key = "thisisadummykey"
CORS(app)

@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def prediction():
    if request.method == 'POST':
        img_recived = request.json.get('image')
        decodeImage(img_recived, 'input_image.jpg')
        
        obj = PredictionService('input_image.jpg')
        result = obj.predict() # resutl a dict as {'image': prediction}
        
        return jsonify(result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1919, debug=True)