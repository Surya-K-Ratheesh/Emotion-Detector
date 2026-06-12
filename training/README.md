# Emotion Detector - Model Training & Conversion

This directory contains the Python scripts to preprocess the FER2013 dataset, train a deep Convolutional Neural Network (CNN) in Keras, and convert the trained model to TensorFlow.js format.

## Setup Instructions

### 1. Create a Virtual Environment
We recommend using a Python virtual environment (Python 3.8 - 3.10 is recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries, including TensorFlow and the TensorFlow.js converter:

```bash
pip install -r requirements.txt
```

---

## Dataset Acquisition

The model is trained on the **FER2013** dataset (Facial Expression Recognition 2013).
1. Download `fer2013.csv` from [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013).
2. Place the `fer2013.csv` file directly in this `training/` directory.

> [!NOTE]
> If you run `train.py` without `fer2013.csv` present, it will automatically generate a **synthetic dummy dataset** so that you can verify the end-to-end training and conversion pipeline immediately.

---

## Training and Converting the Model

### 1. Train the Keras Model
Run the training script to load, preprocess, augment the dataset, and train the Keras CNN. The script automatically saves the model with the best validation accuracy to `emotion_model.h5`.

```bash
python train.py
```

### 2. Convert to TensorFlow.js
Run the conversion script to export the trained `.h5` model into the JSON/binary weights format expected by the browser. By default, it will export files directly into the Next.js static directory (`../web/public/models/emotion`).

```bash
python convert.py
```

This generates:
- `model.json`: The model topology/definition.
- `group1-shard1of1.bin` (and potentially other shards): The binary weights files.
