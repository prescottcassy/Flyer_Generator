import { Routes, Route, Link } from 'react-router-dom';
import Generate from './pages/Generate.jsx';
import Upload from './pages/Upload.jsx';
import Gallery from './pages/Gallery.jsx';
import './App.css';


function App() {
  return (
    <>
      <h1>Create Your Perfect Flyer</h1>
      <nav>
        <Link to="/">Generate</Link> | 
        <Link to="/upload">Upload</Link> | 
        <Link to="/gallery">Gallery</Link>
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
