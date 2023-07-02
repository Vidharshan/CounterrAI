import React, { useState } from 'react';
import axios from 'axios';
import ReactAudioPlayer from 'react-audio-player';
//import 'react-audio-player/lib/styles.css';

function App() {
  const [result, setResult] = useState(null);
  const [audiopath, setAudiopath] = useState(null);
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];

    if(!file){
      return;
    }
    console.log(file)
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    console.log(response.data.audiopath);
    setAudiopath('http://127.0.0.1:8080/data/Counters%20Audio/'+response.data.audiopath)
    setResult(response.data.result);
  };

  return (
    <div>
      <h1>Audio Classifier</h1>
      <input type="file" accept="audio/*" onChange={handleFileUpload} />
      <p>The audio file was classified as {result}</p>
      <p>Counter:{audiopath}</p>
      <ReactAudioPlayer src={audiopath} controls />

    </div>
  );
}

export default App;