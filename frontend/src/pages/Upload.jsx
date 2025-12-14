import { useState } from 'react';
import { uploadImage } from '../api';

export default function Upload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState('');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState(null);

  function onFileChange(e) {
    const f = e.target.files?.[0];
    setFile(f || null);
    if (f) setPreview(URL.createObjectURL(f));
    else setPreview(null);
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return setMsg('Select a file');
    setLoading(true);
    setMsg(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('title', title);
      fd.append('tags', tags);
      const res = await uploadImage(fd);
      if (res.ok) setMsg('Upload successful');
      else setMsg(res.error || 'Upload failed');
    } catch (err) {
      setMsg(err.message || 'Upload error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Upload Image</h2>
      <form onSubmit={handleUpload}>
        <input type="file" accept="image/*" onChange={onFileChange} />
        {preview && <img src={preview} alt="preview" style={{ maxWidth: 200 }} />}
        <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Title" />
        <input value={tags} onChange={e => setTags(e.target.value)} placeholder="comma,separated,tags" />
        <button disabled={loading}>{loading ? 'Uploading...' : 'Upload'}</button>
      </form>
      {msg && <div>{msg}</div>}
    </div>
  );
}