
---

# Emotion Recognition using EfficientNet

This project uses the **EfficientNetV2B2** architecture for real-time emotion detection from facial expressions. The model is trained on the **Face Expression Recognition (FER)** dataset.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Real-time Emotion Detection](#real-time-emotion-detection)
- [Improving Model Accuracy](#improving-model-accuracy)
- [Notes](#notes)
- [License](#license)

---

## Installation

1. **Download the complete project** from Google Drive:
   - [Download Here](https://drive.google.com/file/d/10gSP8h-uDKh9_0uyjzAieosodZMen3ME/view?usp=drive_link)

2. Clone the repository:
   ```bash
   git clone https://github.com/Chorko/Emotion-recognition-using-efficientnet.git
   cd Emotion-recognition-using-efficientnet
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up a virtual environment (preferred)**:
   Setting up a virtual environment helps in managing dependencies and avoiding conflicts with system-installed packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   .\venv\Scripts\activate   # For Windows
   ```

5. Install [Git LFS](https://git-lfs.github.com/) if needed for handling large files:
   ```bash
   git lfs install
   ```

---

## Dataset

This project uses the **Face Expression Recognition (FER)** dataset, which contains images of faces classified into 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

### Download the Dataset

The dataset is **not** included in this repository. You need to download it manually:

- [Download FER dataset from Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

### Organizing the Dataset

After downloading the dataset, install it by placing it inside the `FER/` directory as follows:
```
FER/
│
├── images/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── validation/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
```

---

## Model Training

To train the model using **EfficientNetV2B2**, run the `emotion_classification_efficientnet.py` script.

### Training on Google Colab
It is recommended to use **Google Colab** with a **T4 GPU** for training due to the large dataset size and high computation requirements. You can upload this repository and the dataset to your Colab environment.

1. Adjust the paths to the dataset and save directories within the training script as necessary to match your Colab file system.
2. Train the model:
   ```bash
   python emotion_classification_efficientnet.py
   ```

---

## Real-time Emotion Detection

For real-time emotion detection using a webcam, run the `main.py` script. This uses OpenCV and a pre-trained model for detecting faces and predicting emotions.

1. First, make sure the Haar Cascade file for face detection is available:
   ```bash
   haarcascade_frontalface_default.xml
   ```

2. Then, run the script:
   ```bash
   python main.py
   ```

### Using the Pre-trained Model

The pre-trained model `best_model.keras` is **not** included in this repository. You need to train the model or download it separately and place it in the `models/` directory:
```
models/
└── best_model.keras
```

---

## Improving Model Accuracy

To improve model accuracy, consider experimenting with the following:

1. **Learning Rate**:
   - Try lowering or increasing the learning rate. A smaller learning rate can help the model converge more precisely but may take longer.
   - Example: 
     ```python
     learning_rate = 0.001  # or lower like 0.00005
     ```

2. **Increase Number of Epochs**:
   - Training for more epochs can allow the model to learn better features from the data.
   - Example:
     ```python
     epochs = 30  # or 50 for longer training
     ```

3. **Early Stopping (min_delta)**:
   - Adjust the `min_delta` parameter in early stopping to prevent the model from stopping too early when improvements are minimal.
   - Example:
     ```python
     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6)
     ```

4. **Dropout Rates**:
   - Modify the dropout rates to prevent overfitting.
   - Example:
     ```python
     dropout_rate = 0.3  # or increase to 0.4
     ```

### Faster Training with Google Colab

To speed up training, make sure to use Google Colab’s **T4 GPU**. You can enable the GPU runtime in Colab by navigating to **Runtime** > **Change runtime type** > **GPU**.

---

## Notes

- **Trained model and dataset are excluded**: 
  The `FER` dataset and `best_model.keras` file are not included in this repository due to size constraints. You can add these manually following the instructions above.

- **Using `.gitignore`**:
  The `.gitignore` file has been set to exclude the following:
  - The FER dataset (`FER/`)
  - The trained model (`models/best_model.keras`)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### Project Structure

```
Emotion_recognition_using_efficientnet/
│
├── emotion_classification_efficientnet.py  # Script for training EfficientNet on FER dataset
├── main.py                                 # Script for real-time emotion detection using webcam
├── haarcascade_frontalface_default.xml     # Haar Cascade file for face detection
│
├── FER/                                    # Folder containing FER dataset (excluded in .gitignore)
│   └── images/
│       ├── train/
│       └── validation/
│
├── models/                                 # Folder for storing trained models (excluded in .gitignore)
│   └── best_model.keras                    # Trained EfficientNet model (optional)
│
├── README.md                               # Project documentation and instructions
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore file
└── .gitattributes                          # Git attributes for handling text and large files
```

---

![alt text](<Screenshot 2024-09-14 202334.png>)