const fs = require('fs');
const path = require('path');
const https = require('https');
const tf = require('@tensorflow/tfjs');

const FACEAPI_DIR = path.join(__dirname, '..', 'public', 'models', 'faceapi');
const EMOTION_DIR = path.join(__dirname, '..', 'public', 'models', 'emotion');

const FACEAPI_FILES = [
  'tiny_face_detector_model-weights_manifest.json',
  'tiny_face_detector_model-shard1'
];

const FACEAPI_BASE_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// Helper function to download files
function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download: ${response.statusCode} for ${url}`));
        return;
      }
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function setupFaceApi() {
  console.log('--- Setting up Face-API models ---');
  fs.mkdirSync(FACEAPI_DIR, { recursive: true });
  
  for (const filename of FACEAPI_FILES) {
    const destPath = path.join(FACEAPI_DIR, filename);
    if (fs.existsSync(destPath)) {
      console.log(`[FaceAPI] ${filename} already exists, skipping download.`);
      continue;
    }
    
    const url = FACEAPI_BASE_URL + filename;
    console.log(`[FaceAPI] Downloading ${filename}...`);
    try {
      await downloadFile(url, destPath);
      console.log(`[FaceAPI] Downloaded ${filename} successfully.`);
    } catch (err) {
      console.error(`[FaceAPI] Error downloading ${filename}:`, err.message);
      process.exit(1);
    }
  }
}

async function setupEmotionModel() {
  console.log('\n--- Generating Mock TFJS Emotion Model ---');
  fs.mkdirSync(EMOTION_DIR, { recursive: true });
  
  const manifestPath = path.join(EMOTION_DIR, 'model.json');
  if (fs.existsSync(manifestPath)) {
    console.log('[Emotion] model.json already exists. Skipping dummy model generation.');
    return;
  }

  console.log('[Emotion] Creating a sequential model...');
  
  const model = tf.sequential();
  
  // Input: 48x48x1 grayscale image
  model.add(tf.layers.conv2d({
    inputShape: [48, 48, 1],
    filters: 16,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 7, activation: 'softmax' })); // 7 emotions

  console.log('[Emotion] Saving model utilizing custom Node IOHandler...');
  
  // Custom IOHandler to avoid dependency on @tensorflow/tfjs-node
  const customIO = {
    save: async (modelArtifacts) => {
      // 1. Write the weights manifest and topology (model.json)
      const weightsManifest = {
        weightsManifest: [
          {
            paths: ['./group1-shard1of1.bin'],
            weights: modelArtifacts.weightSpecs
          }
        ]
      };
      
      const modelJson = {
        format: 'layers-model',
        generatedBy: 'TensorFlow.js v' + tf.version.tfjs,
        convertedBy: null,
        modelTopology: modelArtifacts.modelTopology,
        weightsManifest: weightsManifest.weightsManifest
      };
      
      fs.writeFileSync(
        path.join(EMOTION_DIR, 'model.json'), 
        JSON.stringify(modelJson, null, 2)
      );
      
      // 2. Write the binary weights
      if (modelArtifacts.weightData) {
        const buffer = Buffer.from(modelArtifacts.weightData);
        fs.writeFileSync(path.join(EMOTION_DIR, 'group1-shard1of1.bin'), buffer);
      }
      
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }
  };

  try {
    await model.save(customIO);
    console.log('[Emotion] Dummy model generated successfully in web/public/models/emotion/');
  } catch (err) {
    console.error('[Emotion] Error saving dummy model:', err);
  }
}

async function run() {
  await setupFaceApi();
  await setupEmotionModel();
  console.log('\nSetup completed successfully!');
}

run();
