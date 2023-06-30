import React, { useState } from 'react';
import axios from 'axios';


function App() {
  const [result, setResult] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    setResult(response.data.result);
  };

  return (
    <div>
      <h1>Audio Classifier</h1>
      <input type="file" name = "audio" accept="audio/*" onChange={handleFileUpload} />
      { <p>The audio file was classified as {result}</p>}
    </div>
  );
}

export default App;