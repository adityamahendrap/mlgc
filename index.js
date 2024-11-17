const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const { Firestore } = require('@google-cloud/firestore');
const cors = require('cors');

const app = express();

const PORT = process.env.PORT || 3000;
const MODEL_PATH = 'https://storage.googleapis.com/submissionmlgc-adityap-gcs/model.json';
const MAX_FILE_SIZE = 1000000; // 1MB file size limit
const FIRESTORE_COLLECTION = 'predictions';

const db = new Firestore({
    projectId: 'submissionmlgc-adityap',
    keyFilename: 'credentials.json', 
});
const predictionsCollection = db.collection(FIRESTORE_COLLECTION);

const upload = multer({
    limits: { fileSize: MAX_FILE_SIZE },

    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only images are allowed.'));
        }
    },
});

let model;

async function loadModel() {
    try {
        model = await tf.loadGraphModel(MODEL_PATH);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

async function savePredictionToFirestore(predictionData) {
    const { id, result, suggestion, createdAt } = predictionData;
    await predictionsCollection.doc(id).set({
        id,
        result,
        suggestion,
        createdAt,
    });
    console.log(`Prediction with ID ${id} saved to Firestore`);
}

app.use(cors());

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!model) {
            return res.status(500).json({
                status: 'fail',
                message: 'Model is not loaded yet',
            });
        }

        const { file } = req;
        if (!file) {
            return res.status(400).json({
                status: 'fail',
                message: 'Image file is required',
            });
        }

        const imageBuffer = file.buffer;
        let imageTensor = tf.node.decodeImage(imageBuffer).resizeBilinear([224, 224]).expandDims(0);
        
        const prediction = model.predict(imageTensor);
        const output = prediction.dataSync()[0];
        const result = output > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        const id = uuidv4();
        const createdAt = new Date().toISOString();

        const predictionData = { id, result, suggestion, createdAt };
        await savePredictionToFirestore(predictionData);

        res.status(201).json({
            status: 'success',
            message: 'Model is predicted successfully',
            data: predictionData,
        });
    } catch (error) {
	console.error(error);

        res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
        });
    }
});

app.get('/predict/histories', async (req, res) => {
    try {
        const snapshot = await predictionsCollection.get();
        const data = snapshot.docs.map(doc => {
            const docData = doc.data();
            return {
                id: docData.id,
                history: {
                    result: docData.result,
                    createdAt: docData.createdAt,
                    suggestion: docData.suggestion,
                    id: docData.id
                }
            };
        });

        res.status(200).json({
            status: 'success',
            data: data
        });
    } catch (error) {
        console.error('Error fetching prediction histories:', error);
        res.status(500).json({
            status: 'fail',
            message: 'Error fetching prediction histories'
        });
    }
});

app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
            status: 'fail',
            message: `Payload content length greater than maximum allowed: ${MAX_FILE_SIZE}`,
        });
    }
    next(err);
});


app.listen(PORT, async () => {
    console.log(`Server is running on port ${PORT}`);
    await loadModel();
});
