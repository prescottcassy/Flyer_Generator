const admin = require("firebase-admin");
const path = require("path");
const fs = require("fs");

// This module initializes Firebase admin SDK if a service account is provided via
// the FIREBASE_SERVICE_ACCOUNT environment variable (recommended). It fails
// gracefully and exports an empty object if Firebase isn't configured.

let exportsObj = {};
try {
    const saPath = process.env.FIREBASE_SERVICE_ACCOUNT || path.resolve(__dirname, 'serviceAccount.json');
    if (!fs.existsSync(saPath)) {
        throw new Error(`Service account not found at ${saPath}. Set FIREBASE_SERVICE_ACCOUNT to the JSON key path to enable Firebase uploads.`);
    }

    const serviceAccount = require(saPath);

    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
        storageBucket: process.env.FIREBASE_STORAGE_BUCKET || "brandbeastcassycormier.appspot.com"
    });

    const db = admin.firestore();
    const bucket = admin.storage().bucket();
    exportsObj = { admin, db, bucket };
    console.log('Firebase admin initialized successfully.');
} catch (err) {
    console.warn('Firebase admin not initialized:', err.message);
}

module.exports = exportsObj;
