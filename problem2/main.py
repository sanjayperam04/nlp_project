"""
Sentiment Classification for Airline Reviews
Problem 2 - Binary classification using ML
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import re


class SentimentClassifier:
    """Classifier for airline review sentiment analysis"""
    
    def __init__(self, use_ensemble=True):
        # TF-IDF vectorizer setup
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True,
            strip_accents='unicode',
            norm='l2'
        )
        
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            # Using ensemble for better results
            lr_model = LogisticRegression(
                max_iter=2000,
                random_state=42,
                C=2.5,
                solver='saga',
                class_weight='balanced',
                penalty='l2'
            )
            
            nb_model = MultinomialNB(alpha=0.1)
            
            svm_model = LinearSVC(
                max_iter=2000,
                random_state=42,
                C=1.5,
                class_weight='balanced',
                dual=False
            )
            
            self.model = VotingClassifier(
                estimators=[
                    ('lr', lr_model),
                    ('nb', nb_model),
                    ('svc', svm_model)
                ],
                voting='hard',
                n_jobs=-1
            )
        else:
            self.model = LogisticRegression(
                max_iter=2000,
                random_state=42,
                C=2.5,
                solver='saga',
                class_weight='balanced',
                penalty='l2'
            )
        
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        
        # Fix contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for old, new in contractions.items():
            text = text.replace(old, new)
        
        # Keep sentiment indicators
        text = text.replace('!', ' exclamation ')
        text = text.replace('?', ' question ')
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep only letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Clean up spaces
        text = ' '.join(text.split())
        
        # Remove short words
        text = ' '.join([w for w in text.split() if len(w) > 1])
        
        return text


def train_sentiment_model(training_data):
    """Train sentiment model on provided data"""
    if not training_data:
        raise ValueError("Training data cannot be empty")
    
    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]
    
    # Create classifier with ensemble
    classifier = SentimentClassifier(use_ensemble=True)
    
    # Clean all texts
    processed_texts = []
    for text in texts:
        cleaned = classifier.preprocess_text(text)
        if cleaned.strip():
            processed_texts.append(cleaned)
        else:
            processed_texts.append("empty review")
    
    # Convert to features
    X = classifier.vectorizer.fit_transform(processed_texts)
    
    # Convert labels to 0/1
    y = np.array([1 if label.lower() == "positive" else 0 for label in labels])
    
    # Train
    classifier.model.fit(X, y)
    classifier.is_trained = True
    
    print(f"Model trained on {len(training_data)} samples")
    print(f"Vocabulary size: {len(classifier.vectorizer.vocabulary_)}")
    if classifier.use_ensemble:
        print("Using ensemble (LR + NB + SVM)")
    
    return classifier


def predict_sentiment(model, new_text):
    """Predict sentiment for new text"""
    if not model.is_trained:
        raise ValueError("Model not trained yet")
    
    cleaned = model.preprocess_text(new_text)
    
    if not cleaned.strip():
        cleaned = "empty review"
    
    X = model.vectorizer.transform([cleaned])
    prediction = model.model.predict(X)[0]
    
    return "positive" if prediction == 1 else "negative"


def load_dataset_from_csv(filepath):
    """
    Load training data from CSV file.
    Specifically designed for the 2026 airline dataset with columns:
    - Review: Customer review text
    - Recommended: yes/no (used as sentiment label)
    - OverallScore: 1-10 rating
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        List of (text, label) tuples
    """
    try:
        df = pd.read_csv(filepath)
        
        print(f"Dataset loaded: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        training_data = []
        
        # Check for the 2026 dataset structure
        if 'Review' in df.columns and 'Recommended' in df.columns:
            print("Using 'Review' and 'Recommended' columns from 2026 dataset")
            
            for _, row in df.iterrows():
                # Skip rows with missing review text
                if pd.isna(row['Review']) or str(row['Review']).strip() == '':
                    continue
                
                text = str(row['Review']).strip()
                recommended = str(row['Recommended']).lower().strip()
                
                # Convert 'yes'/'no' to positive/negative
                if recommended == 'yes':
                    label = "positive"
                elif recommended == 'no':
                    label = "negative"
                else:
                    # Skip if not clear yes/no
                    continue
                
                training_data.append((text, label))
        
        # Fallback: Try OverallScore if Recommended not available
        elif 'Review' in df.columns and 'OverallScore' in df.columns:
            print("Using 'Review' and 'OverallScore' columns")
            
            for _, row in df.iterrows():
                if pd.isna(row['Review']) or str(row['Review']).strip() == '':
                    continue
                
                text = str(row['Review']).strip()
                score = row['OverallScore']
                
                # Skip if score is NaN
                if pd.isna(score):
                    continue
                
                # Convert score to sentiment (7+ positive, 4- negative, skip 5-6)
                if score >= 7:
                    label = "positive"
                elif score <= 4:
                    label = "negative"
                else:
                    continue
                
                training_data.append((text, label))
        
        # Generic fallback
        else:
            text_col = None
            label_col = None
            
            for col in df.columns:
                if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                    text_col = col
                if 'recommend' in col.lower() or 'sentiment' in col.lower() or 'rating' in col.lower():
                    label_col = col
            
            if text_col and label_col:
                print(f"Using columns: {text_col} and {label_col}")
                
                for _, row in df.iterrows():
                    if pd.isna(row[text_col]) or str(row[text_col]).strip() == '':
                        continue
                    
                    text = str(row[text_col]).strip()
                    label_value = row[label_col]
                    
                    # Handle different label formats
                    if isinstance(label_value, str):
                        label_value = label_value.lower().strip()
                        if label_value in ['yes', 'positive', 'pos']:
                            label = "positive"
                        elif label_value in ['no', 'negative', 'neg']:
                            label = "negative"
                        else:
                            continue
                    elif isinstance(label_value, (int, float)):
                        if pd.isna(label_value):
                            continue
                        label = "positive" if label_value >= 7 else "negative"
                    else:
                        continue
                    
                    training_data.append((text, label))
        
        print(f"Loaded {len(training_data)} valid samples")
        
        # Show distribution
        if training_data:
            pos_count = sum(1 for _, label in training_data if label == "positive")
            neg_count = len(training_data) - pos_count
            print(f"Distribution: {pos_count} positive, {neg_count} negative")
        
        return training_data
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return []


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data and print comprehensive metrics.
    
    Args:
        model: Trained classifier
        test_data: List of (text, label) tuples for testing
    """
    texts = [item[0] for item in test_data]
    true_labels = [item[1] for item in test_data]
    
    predictions = [predict_sentiment(model, text) for text in texts]
    
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions, labels=["negative", "positive"])
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "=" * 80)
    print("Model Evaluation Results")
    print("=" * 80)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {conf_matrix[0][0]:6d}              {conf_matrix[0][1]:6d}")
    print(f"Actual Positive        {conf_matrix[1][0]:6d}              {conf_matrix[1][1]:6d}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Positive Class - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}")
    print(f"  Negative Class - Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))


def test_system():
    """Run tests"""
    print("=" * 80)
    print("Testing Sentiment Classification")
    print("=" * 80)
    
    # Some sample data for testing
    sample_data = [
        ("The flight was on time, and the staff was friendly.", "positive"),
        ("I had to wait 3 hours due to a delay. Terrible!", "negative"),
        ("Great legroom and comfortable seats.", "positive"),
        ("Lost my luggage, extremely upset about this.", "negative"),
        ("Check-in was smooth, no issues at all.", "positive"),
        ("Worst airline experience ever. Never flying again.", "negative"),
        ("Amazing service and delicious food.", "positive"),
        ("Rude staff and dirty cabin.", "negative"),
        ("Perfect flight, highly recommend!", "positive"),
        ("Cancelled my flight without notice.", "negative"),
        ("Excellent customer service and on-time departure.", "positive"),
        ("Uncomfortable seats and poor entertainment.", "negative"),
        ("Best airline I've ever flown with.", "positive"),
        ("Overpriced and disappointing experience.", "negative"),
        ("Smooth boarding process and friendly crew.", "positive"),
    ]
    
    print(f"\nTraining model with {len(sample_data)} samples...")
    model = train_sentiment_model(sample_data)
    print("Done!")
    
    # Test it out
    test_cases = [
        "The seats were comfortable and service was great!",
        "They lost my baggage and were very unhelpful!",
        "Nothing special, just an average flight.",
        "I love this airline",
        "Worst experience ever",
        "The crew was professional and attentive",
        "Delayed for hours with no explanation"
    ]
    
    print("\n" + "=" * 80)
    print("Test Predictions")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        pred = predict_sentiment(model, text)
        print(f"\n{i}. \"{text}\"")
        print(f"   -> {pred}")
    
    # Load actual dataset
    print("\n" + "=" * 80)
    print("Loading CSV dataset...")
    print("=" * 80)
    
    data = load_dataset_from_csv("airline_sentiment_data.csv")
    
    if data and len(data) > 10:
        print(f"\nGot {len(data)} samples from CSV")
        
        # Split data
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Training on {len(train_data)} samples...")
        model = train_sentiment_model(train_data)
        print("Training complete!")
        
        # Check accuracy
        evaluate_model(model, test_data)
        
        # Save it
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("\nModel saved")
    else:
        print("\nCouldn't load CSV data")


if __name__ == "__main__":
    test_system()
