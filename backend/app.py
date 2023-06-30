from flask import Flask
from flask import request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os
import numpy as np
import subprocess
import os
import pandas as pd
import librosa
import glob 

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Vanakko! Welcome to CounterAI</h1>'

@app.route("/predict", methods=['POST'])
def predict():
    Emotions = pd.read_csv('/Users/pavanadi/Desktop/Pavan/CounterrAI/backend/saved_model/emotion.csv')
    Y = Emotions['labels'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    model_path = '/Users/pavanadi/Desktop/Pavan/CounterrAI/backend/saved_model/cnnmodel.h5'
    model = load_model(model_path)
    model.summary()
    input_data = request.files['file']
    filename =  input_data.filename
    input_data.save(secure_filename(filename))
    f = "/Users/pavanadi/Desktop/Pavan/CounterrAI/backend/" + filename
   
    X, sample_rate = librosa.load(f,duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    twodim= np.expand_dims(livedf2, axis=1)
    output = model.predict(twodim)
    output = (encoder.inverse_transform((output)))
    s = pd.Series(output[0])
    mode = s.mode()[0]
    print(mode)

    return "hiii"


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
    with open("/Users/pavanadi/Desktop/Pavan/CounterrAI/backend/saved_model/cnnmodel.h5", "rb") as f:
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
    app.config['UPLOAD_FOLDER'] = '/Users/pavanadi/Desktop/Pavan/CounterrAI/data/uploads'
    app.run(debug=True, port=5000)
