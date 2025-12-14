import { useEffect, useState } from 'react';
import { listImages } from '../api';

export default function Gallery() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    listImages().then(data => {
      if (!alive) return;
      setItems(Array.isArray(data) ? data : []);
      setLoading(false);
    }).catch(err => {
      if (!alive) return;
      setError(err.message || 'Failed to load');
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  if (loading) return <div>Loading gallery…</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;

  return (
    <div>
      <h2>Gallery</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: 12 }}>
        {items.map(it => (
          <div key={it.id} style={{ border: '1px solid #ddd', padding: 8 }}>
            <img src={it.url} alt={it.title} style={{ width: '100%', height: 120, objectFit: 'cover' }} />
            <div>{it.title}</div>
            <small>{it.tags}</small>
          </div>
        ))}
      </div>
    </div>
  );
}