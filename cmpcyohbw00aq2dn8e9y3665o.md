---
title: "Building CivicAI : Part 2: Training the ML Pipeline That Powers It All"
seoTitle: "Training NLP & Random Forest Models for Real-World AI System"
seoDescription: "How I built two ML models from scratch, a TF-IDF text classifier and Random Forest priority scorer  and serialized them for production."
datePublished: 2026-05-19T18:23:31.776Z
cuid: cmpcyohbw00aq2dn8e9y3665o
slug: building-civicai-part-2-training-the-ml-pipeline-that-powers-it-all
cover: https://cdn.hashnode.com/uploads/covers/67fef31e0e04698a213cfba5/6f016be2-28fa-44b1-9bfc-debeebab076d.jpg
tags: ai, machine-learning, backend, frontend-development, ml, pipeline, logistic-regression, random-forest

---

*Series: Building an AI-Powered Municipal Complaint Management System from Scratch*

* * *

## Where We Left Off

In Part 1, we built our dataset from scratch — growing it from 200 raw records to a clean, label-rich 1,100-row training set with engineered features like severity scores, area importance, and priority labels.

Now the data is ready. Time to build the models.

In this part, we train **two separate ML models**:

1.  **Category Model** : Given a complaint description in plain text, what type of issue is it? (pothole, graffiti, water leak, etc.)
    
2.  **Priority Model** : Given the features of a complaint, how urgent is it? (low, medium, high, critical)
    

These two models have completely different inputs, different algorithms, and serve different purposes , but together they form the intelligence layer of CivicAI.

* * *

## The Imports

```python
import pandas as pd
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import joblib
```

Every import here is intentional. Notice we're using both `LogisticRegression` and `RandomForestClassifier` that's because our two models solve fundamentally different problems and need different algorithms. More on that shortly.

* * *

## Part 1 : The Category Classifier (NLP Model)

### What Are We Solving?

A citizen submits: *"There is a large pothole causing traffic issues on the main road."*

The system needs to automatically classify this as `pothole` , without a human reading it.

This is a **text classification problem**. The input is raw, unstructured language. The output is one of several issue type categories.

* * *

### Step 1 : Load and Clean the Data

```python
df = pd.read_csv("municipal_training_set_1100.csv")

# Drop rows where the most critical columns are missing
df = df.dropna(subset=["issue_description"])
df = df.dropna(subset=["issue_type"])

# Keep only what we need for this model
df = df[["issue_description", "issue_type"]]
df = df.dropna()
df.columns = ["text", "category"]
```

We only need two columns for the category model: the raw text description and its label. Everything else severity, location, priority is irrelevant here. Keeping the data focused is important.

* * *

### Step 2 : Fix Inconsistent Labels

Real datasets always have this problem. The same issue type gets entered with different formats:

```python
df['category'] = df['category'].replace({
    'water-leak': 'water_leakage',
    'illegal-dumping': 'illegal_dumping'
})
```

`water-leak` and `water_leakage` mean exactly the same thing, but a model treats them as two completely different classes. This kind of label standardization is easy to miss and devastating to model performance if ignored.

* * *

### Step 3 : Text Preprocessing Function

Before any ML model can work with text, it needs to be cleaned and normalized:

```python
def clean_text(text):
    text = text.lower()                          # Normalize case
    text = re.sub(r'[^a-z\s]', '', text)         # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra whitespace
    return text
```

**Why each step matters:**

| Step | Before | After | Reason |
| --- | --- | --- | --- |
| Lowercase | `Pothole`, `pothole`, `POTHOLE` | `pothole` | Same word, same token |
| Remove punctuation | `road!`, `road.`, `road` | `road` | Punctuation adds noise |
| Remove numbers | `3rd street`, `street` | `street` | Numbers rarely carry meaning for classification |
| Strip whitespace | `" pothole "` | `"pothole"` | Prevents ghost tokens |

This function gets passed directly into the TF-IDF vectorizer as the `preprocessor` so it runs automatically on every input, both during training and prediction.

* * *

### Step 4 — Building the Pipeline

This is the most important design decision in the entire NLP model:

```python
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=clean_text)),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])
```

**What is a Pipeline?**

A scikit-learn `Pipeline` chains multiple steps into a single object. When you call `.fit()`, it fits every step in order. When you call `.predict()`, it transforms the input through every step automatically.

```plaintext
Raw Text Input
      ↓
clean_text() — lowercasing, regex cleaning
      ↓
TF-IDF Vectorizer — converts text to numeric feature matrix
      ↓
Logistic Regression — classifies into issue type
      ↓
Predicted Category
```

**Why this matters for production:** When we serialize this pipeline to a `.pkl` file and load it in FastAPI, a single `model.predict(["some complaint text"])` call runs the entire chain. No need to manually clean text or vectorize before calling the model. The pipeline handles it.

* * *

### Understanding TF-IDF

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts text into numbers that a machine learning model can understand.

*   **TF (Term Frequency):** How often does a word appear in this complaint?
    
*   **IDF (Inverse Document Frequency):** How rare is this word across all complaints?
    

The result: common words like *"the"*, *"is"*, *"a"* get low scores. Domain-specific words like *"pothole"*, *"flooding"*, *"graffiti"* get high scores, because they're frequent in specific complaint types but not everywhere.

This is what allows Logistic Regression to learn: *"when I see high TF-IDF scores for 'pothole' and 'road', predict* `pothole`*."*

* * *

### Why Logistic Regression for Text Classification?

Logistic Regression works exceptionally well with TF-IDF features because:

*   TF-IDF produces **high-dimensional sparse vectors** — thousands of features but most are zero
    
*   Logistic Regression handles sparse data efficiently
    
*   It's **fast to train** and easy to interpret
    
*   It generalizes well when you have balanced classes
    

`class_weight='balanced'` tells the model to automatically adjust for any class imbalance — if `graffiti` has fewer examples than `pothole`, the model compensates so it doesn't just learn to always predict the majority class.

`max_iter=1000` gives the optimizer enough iterations to converge on 1,100 rows with multiple classes.

* * *

### Step 5 — Train / Test Split and Training

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42
)

category_pipeline.fit(X_train, y_train)
```

*   **80% training, 20% testing** standard split for this dataset size
    
*   `random_state=42` makes the split reproducible. Run it today or six months from now, same split
    

* * *

### Step 6 — Evaluation

```python
preds = category_pipeline.predict(X_test)
print(classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))
```

Don't just look at accuracy. The `classification_report` gives you **precision, recall, and F1 score per class** which tells you whether the model is struggling with specific issue types.

*   **Precision:** Of all the times it predicted `pothole`, how many were actually `pothole`?
    
*   **Recall:** Of all the actual `pothole` complaints, how many did it correctly catch?
    
*   **F1:** Harmonic mean of precision and recall the balanced single metric
    

* * *

### Step 7 — Save the Model

```python
joblib.dump(category_pipeline, 'category_model.pkl')
print("Model saved!")
```

`joblib` serializes the entire pipeline, TF-IDF vectorizer weights, Logistic Regression coefficients, everything, into a single file. This is what gets loaded inside FastAPI at startup.

### Verify It Works

```python
loaded_model = joblib.load('category_model.pkl')

test_desc = ["garbage not collected since three days"]
prediction = loaded_model.predict(test_desc)
print("Predicted category:", prediction[0])
```

The model correctly predicted the category from a raw description it had never seen, exactly what we need for production.

* * *

## Part 2 : The Priority Classifier (Tabular Model)

### What Are We Solving?

Once we know the issue type, we need to know how urgent it is. But priority isn't determined by text description alone, it depends on multiple features working together:

*   How dangerous is the issue? (`severity_score`)
    
*   How critical is the location? (`area_importance`)
    
*   How many citizens reported it? (`citizen_reports_count`)
    
*   What type of issue is it? (`issue_type`)
    

This is a **tabular classification problem** , structured numeric features in, priority label out. A completely different problem from the NLP model above.

* * *

### Step 1 — Load and Standardize

```python
df_priority = pd.read_csv('municipal_training_set_1100.csv')

df_priority['issue_type'] = df_priority['issue_type'].replace({
    'water-leak': 'water_leakage',
    'illegal-dumping': 'illegal_dumping'
})
```

Same label standardization as before, consistency across both models matters.

* * *

### Step 2 — Encode Issue Type

The `issue_type` column is text (`"pothole"`, `"graffiti"`, etc.) but Random Forest needs numbers. `LabelEncoder` converts each unique string to an integer:

```python
df_priority['issue_type'] = LabelEncoder().fit_transform(df_priority['issue_type'])
```

```plaintext
"graffiti"    → 0
"pothole"     → 1
"water_leak"  → 2
...
```

**Important:** This encoding is applied before training. In production (FastAPI), we apply the same encoding to incoming requests before calling `rf_model.predict()`.

* * *

### Step 3 — Define Features and Target

```python
features = ['severity_score', 'area_importance', 'citizen_reports_count', 'issue_type']
target = 'priority_level'

X = df_priority[features]
y = df_priority[target]
```

These four features were all engineered in Part 1 using our domain formulas. This is the payoff of that work, clean, meaningful features going into the model.

* * *

### Step 4 — Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Same 80/20 split, same `random_state=42` for reproducibility.

* * *

### Step 5 — Train the Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
```

**Why Random Forest for priority classification?**

| Reason | Explanation |
| --- | --- |
| Handles mixed feature scales | `severity_score` is 1–10, `citizen_reports_count` is 1–50 — RF doesn't need normalization |
| Non-linear decision boundaries | Priority isn't a simple linear combination of features |
| Robust to outliers | A few extreme severity scores won't dominate the model |
| Built-in feature importance | After training, RF tells you which features matter most |
| Ensemble of 100 trees | Each tree votes; majority wins — reduces overfitting significantly |

`n_estimators=100` means 100 decision trees vote on every prediction. More trees = more stable predictions, at the cost of slightly slower training (still fast on 1,100 rows).

`class_weight='balanced'` same reason as before. `critical` complaints are rarer than `low` by design. This prevents the model from learning to always predict `low`.

* * *

### Step 6 — Evaluation

```python
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
```

Check precision and recall across all four priority levels. Pay particular attention to `critical` that's the most important class to get right. A missed `critical` complaint in a real municipal system has real consequences.

* * *

### Step 7 — Save the Model

```python
joblib.dump(rf_model, 'priority_model.pkl')
print("Model saved!")
```

Note that unlike the category model, we saved the Random Forest **without** a Pipeline wrapper. That's because the priority model's inputs are already numeric no text transformation needed. In FastAPI, we apply the `LabelEncoder` logic manually before calling `rf_model.predict()`.

* * *

## Two Models, Two Files, One System

After this notebook runs, we have exactly two files:

```plaintext
category_model.pkl   ← TF-IDF + Logistic Regression pipeline
priority_model.pkl   ← Random Forest classifier
```

Here's how they fit together in the full system:

```plaintext
User submits: "Dangerous flooding near hospital road"
                          │
                          ▼
            category_model.pkl
            TF-IDF → Logistic Regression
                          │
                    Predicted: "flooding"
                          │
                          ▼
         Feature Engineering (FastAPI)
         severity=9, area_imp=10, count=3
         issue_type → LabelEncoder → int
                          │
                          ▼
            priority_model.pkl
            Random Forest (100 trees)
                          │
                    Predicted: "critical"
                          │
                          ▼
              API Response to Frontend
              priority_level: "critical"
              priority_score: 9.2
```

* * *

## Key Design Decisions Recap

| Decision | What we chose | Why |
| --- | --- | --- |
| Text model algorithm | Logistic Regression | Works best with sparse TF-IDF vectors |
| Tabular model algorithm | Random Forest | Handles mixed scales, non-linear patterns |
| Text model packaging | sklearn Pipeline | Preprocessing + model in one serializable object |
| Serialization | joblib | Faster and more efficient than pickle for numpy arrays |
| Class imbalance handling | `class_weight='balanced'` | Prevents majority class dominance in both models |
| Reproducibility | `random_state=42` | Same results every run, every machine |

* * *

## What's Next

We have two trained, serialized models sitting as `.pkl` files. In **Part 3**, we'll load these into **FastAPI**, build the `/complaints` endpoint, integrate the Nominatim geocoding API and Overpass API for nearby facilities, and return the full structured response that the Next.js frontend consumes.

The models are trained. Now we ship them.

* * *

*Part of the CivicAI series, building a full-stack ML municipal management system with FastAPI, Next.js, scikit-learn, and Docker.*

* * *

*Found this useful? Drop a reaction or follow for Part 3. Questions about any step? Leave them in the comments.*