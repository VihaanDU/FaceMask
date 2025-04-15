# 😷 Face Mask Detection using CNN & OpenCV

## 📝 Overview

This project is a real-time computer vision system that detects whether people are wearing face masks or not. It uses a **Haar Cascade Classifier** to locate faces in images or video frames, and a **Convolutional Neural Network (CNN)** to classify each face as either:
- ✅ Wearing a mask
- ❌ Not wearing a mask

The goal is to demonstrate how machine learning and image processing can be combined to solve a real-world problem, especially relevant in public health monitoring scenarios like the COVID-19 pandemic.

---

## 📁 Project Structure

```
FaceMask-main/
├── Project_Code.py                   # Main script for real-time webcam-based detection
├── FMDF.py                           # Alternate script for image-based mask detection
├── mask_recog.h5                     # Pre-trained Keras CNN model for mask classification
├── haarcascade_frontalface_alt2.xml # Haar Cascade classifier for detecting human faces
├── image.jpg, image2.png, ...        # Sample images for manual testing
├── *.zip                             # Optional pre-packaged Python dependencies (not required)
└── README.md                         # Project documentation
```

---

## 🧠 How It Works

1. **Face Detection**:
   - Uses OpenCV’s Haar cascade (`haarcascade_frontalface_alt2.xml`) to detect faces in a frame.

2. **Preprocessing**:
   - Each face is resized to 224x224 pixels and normalized using MobileNetV2 preprocessing.

3. **Classification**:
   - The pre-trained CNN model (`mask_recog.h5`) classifies the face as either:
     - **With Mask** 😷
     - **Without Mask** 😐

4. **Output**:
   - The result is displayed with bounding boxes:
     - ✅ Green box for mask
     - ❌ Red box for no mask

---

## 🚀 How to Run the Project

### 🧰 Prerequisites

Make sure you have Python installed and run the following command to install dependencies:

```bash
pip install tensorflow opencv-python numpy
```

### ▶️ Option 1: Run Real-Time Detection

This uses your webcam to detect and classify faces in real time.

```bash
python Project_Code.py
```

Ensure that `mask_recog.h5` and `haarcascade_frontalface_alt2.xml` are in the same directory.

### ▶️ Option 2: Test on Sample Images

You can run detection on the provided image files:

```bash
python FMDF.py
```

Replace the default images with your own for custom tests.

---

## 📸 Sample Output

Each face is identified and labeled with:
- ✅ **MASK** (in green box)
- ❌ **NO MASK** (in red box)

This makes it visually easy to evaluate detection accuracy and performance.

---

## 🧪 Model Training (Optional)

If you wish to retrain the model:

1. Gather a dataset with two categories:
   - `with_mask/`
   - `without_mask/`

2. Train a CNN using TensorFlow/Keras.

3. Save it as:

```python
model.save("mask_recog.h5")
```

---

## ⚠️ Notes

- The `.zip` files (like `numpy-*.zip`, `grpc-*.zip`) are backup offline dependencies — they’re **not needed if you're connected to the internet**.
- For best results, ensure lighting is good and faces are clearly visible in the camera feed.
- Works on CPU but performs faster with GPU acceleration (TensorFlow with CUDA).

---

## Output


## 🙋‍♂️ Author

Created by: *[Your Name Here]*  
For questions or contributions, feel free to reach out or fork the project on GitHub.
