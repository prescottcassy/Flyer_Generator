import { useState } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import * as GenerateModule from './pages/Generate.jsx';
import * as UploadModule from './pages/Upload.jsx';
import * as GalleryModule from './pages/Gallery.jsx';
import './App.css';

function App() {
  const [count, setCount] = useState(0);

  const Generate = GenerateModule.default || GenerateModule.Generate || (() => <div>Generate module unavailable</div>);
  const Upload = UploadModule.default || UploadModule.Upload || (() => <div>Upload module unavailable</div>);
  const Gallery = GalleryModule.default || GalleryModule.Gallery || (() => <div>Gallery module unavailable</div>);

  return (
    <>
      <h1>Create Your Perfect Flyer</h1>
      <nav>
        <Link to="/">Generate</Link> | <Link to="/upload">Upload</Link> | <Link to="/gallery">Gallery</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Generate />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/gallery" element={<Gallery />} />
      </Routes>
    </>
  );
}

export default App;
