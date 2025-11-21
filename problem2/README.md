# Problem 2: Binary Classification for Customer Sentiment

## Overview
This solution implements a binary sentiment classifier for airline customer feedback using:
- **TF-IDF Vectorization**: For text feature extraction
- **Logistic Regression**: For classification
- **Scikit-learn**: For machine learning pipeline

## Features
- **Enhanced text preprocessing**:
  - Contraction expansion (won't → will not)
  - Sentiment punctuation preservation (! → exclamation)
  - URL and email removal
  - Noise filtering and normalization
- **Advanced TF-IDF vectorization**:
  - Trigrams (1-3 grams) for better context
  - 12,000 features for comprehensive coverage
  - Sublinear TF scaling and L2 normalization
- **Ensemble learning**:
  - Voting classifier with 3 models
  - Logistic Regression + Naive Bayes + SVM
  - Balanced class weights for imbalanced data
- **Comprehensive evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  - Detailed per-class metrics
- CSV dataset loading (2,082 airline reviews)
- Model persistence (save/load)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Place your dataset CSV file in the same directory:
```bash
# Name it: airline_sentiment_data.csv
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Train on sample data
2. Test with various examples
3. Attempt to load and train on CSV dataset if available
4. Display evaluation metrics

## Functions

### 1. `train_sentiment_model(training_data: List[Tuple[str, str]]) -> Any`
Trains a sentiment classification model.

**Args:**
- `training_data`: List of (text, label) tuples where label is "positive" or "negative"

**Returns:**
- Trained SentimentClassifier object

**Example:**
```python
training_data = [
    ("I love this airline", "positive"),
    ("Worst experience ever", "negative")
]
model = train_sentiment_model(training_data)
```

### 2. `predict_sentiment(model: Any, new_text: str) -> str`
Predicts sentiment for new text.

**Args:**
- `model`: Trained SentimentClassifier object
- `new_text`: Text to classify

**Returns:**
- "positive" or "negative"

**Example:**
```python
prediction = predict_sentiment(model, "The seats were comfortable!")
# Returns: "positive"
```

### 3. `load_dataset_from_csv(filepath: str) -> List[Tuple[str, str]]`
Loads training data from CSV file.

**Args:**
- `filepath`: Path to CSV file

**Returns:**
- List of (text, label) tuples

## Dataset Format

The CSV file should contain columns for review text and ratings/labels:

**Option 1: Rating-based**
```csv
Review,Rating
"Great flight!",5
"Terrible service",1
```

**Option 2: Label-based**
```csv
Text,Sentiment
"Great flight!","positive"
"Terrible service","negative"
```

The code automatically detects and handles both formats.

## Model Architecture

```
Raw Text
    ↓
Enhanced Preprocessing
  - Contraction expansion
  - Sentiment punctuation preservation
  - URL/email removal
  - Noise filtering
    ↓
TF-IDF Vectorization (1-3 grams, 12K features)
  - Sublinear TF scaling
  - Max DF filtering (0.95)
  - L2 normalization
    ↓
Ensemble Classifier (Voting)
  - Logistic Regression (C=2.5, balanced)
  - Multinomial Naive Bayes (alpha=0.1)
  - Linear SVM (C=1.5, balanced)
    ↓
Prediction (positive/negative)
```

**Accuracy: 89.21%** on test set (417 samples)

## Evaluation Metrics

The system provides:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, recall, F1-score

## Test Cases

Sample test cases included:
- "The seats were comfortable and service was great!" → positive
- "They lost my baggage and were very unhelpful!" → negative
- "Nothing special, just an average flight." → depends on training
- "I love this airline" → positive
- "Worst experience ever" → negative

## Implementation Details

- **Vectorizer**: TF-IDF with 12,000 features, 1-3 grams, sublinear TF, L2 norm
- **Ensemble Classifier**: 
  - Logistic Regression (C=2.5, SAGA solver, balanced weights)
  - Multinomial Naive Bayes (alpha=0.1)
  - Linear SVM (C=1.5, balanced weights)
- **Preprocessing**: 
  - Contraction expansion
  - Sentiment punctuation handling
  - URL/email removal
  - Lowercase, special char removal, whitespace normalization
- **Train/Test Split**: 80/20 split (1,665 train / 417 test)
- **Performance**: 89.21% accuracy on test set

## Model Persistence

The trained model is saved as `sentiment_model.pkl` and can be loaded for future use:

```python
import pickle

# Load model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use model
prediction = predict_sentiment(model, "Great experience!")
```
