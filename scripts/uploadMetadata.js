#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const metadataPath = process.argv[2] || 'metadata.json';

async function main() {
  if (!fs.existsSync(metadataPath)) {
    console.error(`uploadMetadata.js: metadata file not found: ${metadataPath}`);
    process.exit(1);
  }

  console.log(`uploadMetadata.js: received metadata path: ${metadataPath}`);

  let firebase;
  try {
    firebase = require('./firebase-init');
  } catch (err) {
    console.warn('Firebase init unavailable or failed - skipping real upload.');
    console.warn('Error:', err.message);
    console.log('uploadMetadata.js: done (no-op).');
    process.exit(0);
  }

  // If firebase-init didn't initialize (exports empty), skip upload
  if (!firebase || !firebase.db || !firebase.bucket || !firebase.admin) {
    console.warn('Firebase not configured (missing db/bucket/admin). Skipping upload.');
    console.log('uploadMetadata.js: done (no-op).');
    process.exit(0);
  }

  try {
    const raw = fs.readFileSync(metadataPath, 'utf8');
    const data = JSON.parse(raw);
    const itemCount = Array.isArray(data) ? data.length : 1;

    // Build a safe storage destination path
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const dest = `metadata/${timestamp}.json`;

    console.log(`Uploading ${metadataPath} to Storage as ${dest}...`);
    await firebase.bucket.upload(path.resolve(metadataPath), { destination: dest });
    console.log('Upload to Storage complete.');

    // Create a Firestore document describing this upload
    const doc = {
      storagePath: dest,
      itemCount: itemCount,
      uploadedAt: firebase.admin.firestore.FieldValue.serverTimestamp(),
    };

    const writeRes = await firebase.db.collection('metadataUploads').add(doc);
    console.log(`Firestore doc written: ${writeRes.id}`);

    console.log('uploadMetadata.js: done (success).');
    process.exit(0);
  } catch (err) {
    console.error('uploadMetadata.js: failed:', err);
    process.exit(2);
  }
}

main();
