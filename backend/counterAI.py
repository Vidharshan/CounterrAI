from flask import Flask
from flask import request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import numpy as np
import os
import pandas as pd
import librosa
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return '<h1>Vanakko! Welcome to CounterAI</h1>'

@app.route("/predict", methods=['POST'])
def predict():
    Emotions = pd.read_csv('D:/CounterrAI/backend/saved_model/emotion.csv')
    Y = Emotions['labels'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    model_path = 'D:/CounterrAI/backend/saved_model/cnnmodel.h5'
    model = load_model(model_path)
    # model.summary()
    input_data = request.files['file']
    filename =  input_data.filename
    f = "D:/CounterrAI/data/uploads/" + secure_filename(filename)
    input_data.save(f)
    
    X, sample_rate = librosa.load(f,duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    twodim= np.expand_dims(featurelive, axis=1)
    output = model.predict(twodim)
    print(output)
    output = (encoder.inverse_transform((output)))
    print(output)
    s = pd.Series(output[0])
    print(s)
    mode = s.mode()[0]
    print(mode)
    counterlist=[]
    audiofiles = pd.read_csv('D:/CounterrAI/data/counter-emo map.csv', index_col=False)
    for index,row in audiofiles.iterrows():
        print(row['File Name'], "dpjoo",row['Emotion'])
        if row['Emotion']==mode:
            counterlist.append(row['File Name'])
    print(counterlist)
    if(len(counterlist)):
        audiopath=counterlist[int(random.random())%len(counterlist)]
    else:
        audiopath=audiofiles['File Name'][int(random.random())%len(audiofiles['File Name'])]
    
    return jsonify({'result': mode,
                    'audiopath': audiopath})

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'D:/CounterrAI/data/uploads/'
    app.run(debug=True, port=5000)
