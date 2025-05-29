# 🛡️ AI-based Email Spam Detector (Tkinter)

An AI-driven desktop application that detects spam emails using Natural Language Processing (NLP) and a Naive Bayes classifier. Designed with a modern GUI using Python’s Tkinter, the tool supports both individual and bulk email classification, real-time prediction, and user-friendly interaction.

---

## 🎯 Project Purpose

With the exponential rise in email usage, spam and phishing emails have become major cybersecurity threats. This project was developed to:

* Apply AI in solving real-world cybersecurity challenges
* Gain hands-on experience with NLP and machine learning
* Create a functional desktop application with a clean user interface
* Integrate traditional Python libraries with modern AI techniques

---

## 🚀 Key Features

* **Spam Detection**: Classifies emails as "Spam" or "Legit" with confidence scores
* **Interactive GUI**: Built using Tkinter with Light and Cyber themes
* **NLP Preprocessing**: Lowercasing, punctuation removal, stopword filtering, and lemmatization
* **Batch Processing**: Classify multiple emails via CSV file upload
* **Logging**: Save classification results with timestamps into a CSV file
* **Theme Toggle**: Easily switch between dark and light UI themes
* **Preview Output**: View cleaned text and classification results with dynamic coloring

---

## ⚙️ How It Works

### 1. Input

* Users can enter an email manually or upload it from a `.txt` file
* For batch classification, a `.csv` file with emails can be uploaded

### 2. Preprocessing

* Email text is cleaned using the following NLP steps:

  * Convert to lowercase
  * Remove punctuation
  * Remove stopwords (using NLTK)
  * Lemmatize words

### 3. Vectorization

* Cleaned text is transformed into numerical vectors using `TfidfVectorizer`

### 4. Prediction

* A trained `Multinomial Naive Bayes` model predicts whether the email is spam
* A confidence percentage is calculated and shown to the user

### 5. Output

* Results are displayed in the GUI
* Optional: Save prediction to log file (`email_log.csv`)
* For CSV input: predictions are appended to a new file with labels and confidence values

---

## 🛠️ Tech Stack

| Library    | Functionality                           |
| ---------- | --------------------------------------- |
| `tkinter`  | GUI development                         |
| `nltk`     | Stopword removal, lemmatization         |
| `sklearn`  | TF-IDF vectorization, Naive Bayes model |
| `pandas`   | CSV file handling                       |
| `csv`      | Exporting predictions to CSV            |
| `datetime` | Timestamp logging                       |
| `string`   | Punctuation removal                     |

---


