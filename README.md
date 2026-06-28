#  Brain Tumor Classification Using CNN

A deep learning project that classifies brain MRI scans into four categories using a custom Convolutional Neural Network (CNN), achieving **92.5% test accuracy**.

---

##  Problem Statement

Brain tumors are among the most life-threatening conditions, and early detection is critical for effective treatment. This project automates the classification of brain MRI images into four categories using deep learning, reducing the dependency on manual diagnosis.

---

##  Dataset

- **Source:** [Brain Tumor Classification (MRI) – Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes:**
  | Class | Training Images (after augmentation) |
  |---|---|
  | Glioma Tumor | 826 |
  | Meningioma Tumor | 822 |
  | Pituitary Tumor | 827 |
  | No Tumor | 801 (augmented from 395) |
- **Total samples used:** 3,276
- **Image size:** Resized to 48×48 pixels

---

##  Methodology

### 1. Data Preprocessing
- Images resized to **48×48** and normalized to `[0, 1]`
- **Data augmentation** applied to the underrepresented `no_tumor` class using `ImageDataGenerator`:
  - Rotation (±20°), zoom (10%), horizontal flip

### 2. Model Architecture

```
Input (48×48×3)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten
→ Dense(128) + Dropout(0.2)
→ Dense(64) + Dropout(0.2)
→ Dense(4, softmax)
```

- **Total Parameters:** 691,716
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy

### 3. Training
- **Train/Test Split:** 80/20
- **Epochs:** 15
- **Final Training Accuracy:** ~98%
- **Final Validation Accuracy:** ~92.5%

---

##  Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma Tumor | 0.98 | 0.97 | 0.97 |
| Meningioma Tumor | 0.92 | 0.85 | 0.88 |
| No Tumor | 0.89 | 0.92 | 0.90 |
| Pituitary Tumor | 0.92 | 0.98 | 0.95 |
| **Overall Accuracy** | | | **0.93** |

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Model building & training |
| OpenCV | Image reading & resizing |
| scikit-learn | Train-test split, metrics |
| NumPy | Array operations |
| Matplotlib | Visualization |
| opendatasets | Kaggle dataset download |

---

##  Getting Started

### Prerequisites
```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib opendatasets
```

### Run the Project
1. Clone this repository
   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. Open the notebook in Google Colab or Jupyter
3. Provide your Kaggle credentials when prompted to download the dataset
4. Run all cells sequentially

---

##  Project Structure

```
brain-tumor-classification/
│
├── brain_tumor_classification.ipynb   # Main notebook
├── README.md                          # Project documentation
```

---

## Future Improvements

- Use transfer learning (e.g., VGG16, ResNet50) for higher accuracy
- Train on higher resolution images (128×128 or 224×224)
- Deploy as a web app using Flask or Streamlit
- Add Grad-CAM visualizations to highlight tumor regions
---
