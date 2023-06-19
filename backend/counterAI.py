from flask import Flask
from flask import request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
import pickle
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Vanakko! Welcome to CounterAI</h1>'

@app.route("/predict", methods=['POST'])
def predict():
    model_path = 'D:/counterAI/backend/saved_model/cnnmodel.h5'
    model = load_model(model_path)
    model.summary()
    input_data = request.body.get("input/data")
    print(input_data)
    output = model.predict(input_data)
    return jsonify({"output": output})


@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file uploaded'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return jsonify({'filename': filename})

    # Load the Keras model
    with open("D:\counterAI\saved_model\cnnmodel.h5", "rb") as f:
        model = load_model(f)

    # Load the audio file and preprocess it
    # audio, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
    # mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # mfccs = np.mean(mfccs.T, axis=0)
    # input_data = np.expand_dims(mfccs axis=0)

    input_data = request.args.get("input\data")
    
    # Make a prediction using the model
    output = model.predict(input_data)

    return jsonify({'result': output.tolist()})


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'D:/counterAI/data/uploads'
    app.run(debug=True, port=5000)
