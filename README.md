# Sentiment Analysis using CNN (PyTorch)

## Overview

This project implements a **Sentiment Analysis model** using a Convolutional Neural Network (CNN) trained on the IMDB movie reviews dataset.

The goal was not only to build a working model but to **understand model learning behavior**, improve performance through experimentation, and ensure consistency between training and inference.

---

## Features

* Text preprocessing and cleaning
* Custom vocabulary creation and encoding
* CNN-based deep learning model
* Experiment tracking with parameter logging
* Training visualization (loss & accuracy trends)
* Real-time sentiment prediction system
* Handling uncertain predictions

---

## Model Architecture

* **Embedding Layer** (128-dim word vectors)
* **Convolutional Layers** (filters: 2, 3, 4, 5)
* **Max Pooling**
* **Dropout (0.4)** for regularization
* **Fully Connected Layer** for classification

---

## Training Visualization

### Loss Curve

![Loss Curve](assets/loss_curve.png)

Shows how the training loss decreases over epochs, indicating that the model is learning effectively.

### Accuracy Over Runs

![Accuracy](assets/accuracy_runs.png)

Tracks how model accuracy improved across different experiments and parameter tuning.

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix shows that the model correctly classifies most positive and negative reviews, with minor errors in ambiguous cases.

---

## Experiments & Results

| Data Size | Epochs | Accuracy   |
| --------- | ------ | ---------- |
| 10k       | 6      | 0.8155     |
| 20k       | 8      | 0.8347     |
| 20k       | 10     | 0.8462     |
| 20k       | 12     | **0.8530** |

---

## Key Observations

* Increasing dataset size improved generalization
* More convolutional filters improved feature extraction
* Dropout helped reduce overfitting
* Gradient clipping stabilized training
* Accuracy improvements plateaued after ~85% with CNN

---

## Challenges Faced

* **Incorrect predictions due to vocabulary mismatch**
  → Fixed by saving and loading the same vocabulary (`vocab.pkl`)

* **Overfitting at low loss values**
  → Solved using dropout and parameter tuning

* **Uncertain predictions on short/neutral sentences**
  → Handled using confidence threshold

---

## Sample Predictions

```
Input: This movie was amazing and emotional  
Output: Positive (99.87%)

Input: Worst movie ever seen  
Output: Negative (96.93%)

Input: It's too good  
Output: Uncertain (~59%)
```

---

## Tech Stack

* Python
* PyTorch
* Pandas
* Matplotlib
* Scikit-learn

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Train the model

```
python train.py
```

---

### 3. Run prediction app

```
python app.py
```

---

## Project Structure

```
Sentiment-Analysis-CNN/
│
├── train.py
├── app.py
├── model.pth
├── vocab.pkl
├── accuracy_log.txt
├── requirements.txt
├── README.md
│
└── assets/
    ├── loss_curve.png
    ├── accuracy_runs.png
    ├── confusion_matrix.png
```

---

## Key Learnings

* CNN can effectively capture local patterns in text
* Model performance improves significantly with more data
* Tracking experiments is essential in ML workflows
* Consistency between training and inference is critical
* Real-world ML involves debugging, not just training

---

## Future Improvements

* Use pretrained embeddings (GloVe / Word2Vec)
* Implement LSTM / Transformer-based models
* Deploy using Streamlit web app
* Add validation accuracy tracking

---

## Author

Anjali Yadav
