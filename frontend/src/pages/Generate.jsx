import { useState } from 'react';
import { generate } from '../api'; // your central API client

export default function Generate() {
  const [prompt, setPrompt] = useState('');
  const [numSteps, setNumSteps] = useState(50);
  const [loading, setLoading] = useState(false);
  const [imageSrc, setImageSrc] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setImageSrc(null);
    try {
      const res = await generate(prompt, numSteps);
      if (res.image) setImageSrc(`data:image/png;base64,${res.image}`);
      else if (res.url) setImageSrc(res.url);
      else setError('Unexpected response from server');
    } catch (err) {
      setError(err.message || 'Generation failed');
    } finally {
      setLoading(false);
    }
  }

  const handleSave = () => {
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = 'generated.png';
    link.click();
  };

  const handleOpen = () => {
    window.open(imageSrc, '_blank');
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Describe the flyer..." />
        <div className="button-group">
          <button type="submit" disabled={loading || !prompt}>Generate</button>
          {imageSrc && (
            <>
              <button type="button" onClick={handleSave}>Save</button>
              <button type="button" onClick={handleOpen}>Open</button>
            </>
          )}
        </div>
      </form>

      {loading && <div>Generating... (this can take a while)</div>}
      {error && <div className="error">{error}</div>}

      {imageSrc && (
        <div>
          <img src={imageSrc} alt="Generated" style={{ maxWidth: '100%' }} />
        </div>
      )}
    </div>
  );
}
