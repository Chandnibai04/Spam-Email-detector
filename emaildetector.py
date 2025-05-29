import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime
import os
import csv

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Sample data
data = {
    "text": [
        "Win money now!!!",
        "Important update about your account",
        "You’ve been selected for a prize",
        "Hi, how are you?",
        "Can we meet tomorrow?",
        "Congratulations, you won a lottery!",
        "Your invoice is attached",
        "Click this link to claim your reward",
        "Let's catch up soon",
        "Urgent! Account suspended. Verify now."
    ],
    "label": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Clean text
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
model_accuracy = model.score(X_test, y_test)

# ---------------- THEMES ----------------
cyber_theme = {
    "bg": "#0a0f0d",
    "fg": "#00ff99",
    "btn_bg": "#1a1a1a",
    "entry_bg": "#1f1f1f",
    "highlight": "#00ff99"
}

light_theme = {
    "bg": "#f0f0f0",
    "fg": "#0a0f0d",
    "btn_bg": "#e0e0e0",
    "entry_bg": "#ffffff",
    "highlight": "#0a0f0d"
}

current_theme = cyber_theme
theme_state = {"current": "cyber"}

# ---------------- GUI FUNCTIONS ----------------
def apply_theme(theme):
    root.configure(bg=theme['bg'])
    frame.configure(bg=theme['bg'])
    title.configure(bg=theme['bg'], fg=theme['fg'])
    cleaned_label.configure(bg=theme['bg'], fg=theme['fg'])
    result_label.configure(bg=theme['bg'])
    accuracy_label.configure(bg=theme['bg'], fg="gray")
    email_entry.configure(bg=theme['entry_bg'], fg=theme['fg'], insertbackground=theme['fg'])
    cleaned_text_output.configure(bg=theme['entry_bg'], fg=theme['fg'], insertbackground=theme['fg'])
    for btn in [check_button, upload_button, clear_button, theme_button, save_button, csv_button]:
        btn.configure(bg=theme['btn_bg'], fg=theme['highlight'], activebackground=theme['highlight'])

def toggle_theme():
    global current_theme
    if theme_state["current"] == "cyber":
        current_theme = light_theme
        theme_state["current"] = "light"
    else:
        current_theme = cyber_theme
        theme_state["current"] = "cyber"
    apply_theme(current_theme)

def classify_email():
    user_input = email_entry.get("1.0", "end-1c").strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter an email to classify.")
        return

    clean_input = clean_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)[0]
    proba = model.predict_proba(vector_input)[0]
    confidence_spam = proba[1] * 100
    confidence_legit = proba[0] * 100

    # Debugging print statement to track prediction and probability
    print(f"Prediction: {prediction}, Spam Probability: {confidence_spam}, Legit Probability: {confidence_legit}")

    if prediction == 1:
        result_text = (f"🛑 Spam: {confidence_spam:.2f}%\n")
        result_label.config(fg="red")
    else:
        result_text = (f"✅ Legit: {confidence_legit:.2f}%\n")
        result_label.config(fg="green")

    cleaned_text_output.delete("1.0", "end")
    cleaned_text_output.insert("1.0", clean_input)
    result_label.config(text=result_text)

def save_prediction():
    user_input = email_entry.get("1.0", "end-1c").strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter an email to save.")
        return

    clean_input = clean_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)[0]
    proba = model.predict_proba(vector_input)[0]
    confidence = round(max(proba) * 100, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = "email_log.csv"
    file_exists = os.path.exists(filename)

    with open(filename, mode="a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Original Email", "Cleaned Text", "Prediction", "Confidence (%)"])
        writer.writerow([timestamp, user_input, clean_input, "Spam" if prediction == 1 else "Legit", confidence])

def load_email_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            email_entry.delete("1.0", "end")
            email_entry.insert("1.0", content)

def clear_text():
    email_entry.delete("1.0", "end")
    cleaned_text_output.delete("1.0", "end")
    result_label.config(text="")

def process_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        df_csv = pd.read_csv(file_path)
        df_csv = df_csv.dropna(subset=['email'])  # Drop rows with missing emails

        if 'email' not in df_csv.columns:
            messagebox.showerror("Format Error", "CSV must contain a column named 'email'.")
            return

        df_csv['cleaned'] = df_csv['email'].apply(clean_text)
        X_input = vectorizer.transform(df_csv['cleaned'])
        df_csv['prediction'] = model.predict(X_input)
        df_csv['confidence'] = model.predict_proba(X_input).max(axis=1) * 100
        df_csv['label'] = df_csv['prediction'].map({0: "Legit", 1: "Spam"})

        preview = ""
        for idx, row in df_csv.head(5).iterrows():
            color = "🔴" if row['label'] == "Spam" else "🟢"
            preview += f"{color} {row['email'][:50]}...\n   → {row['label']} ({row['confidence']:.2f}%)\n\n"

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if save_path:
            df_csv.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Processed and saved results to:\n{save_path}")

        show_results_popup(preview)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process CSV:\n{str(e)}")

def show_results_popup(content):
    popup = tk.Toplevel(root)
    popup.title("Batch Classification Preview")
    popup.geometry("600x300")
    popup.config(bg=current_theme['bg'])

    text_widget = tk.Text(popup, bg=current_theme['entry_bg'], fg=current_theme['fg'], font=("Consolas", 11))
    text_widget.insert("1.0", content)
    text_widget.config(state="disabled")
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)

# ---------------- UI SETUP ----------------
root = tk.Tk()
root.title("🛡️ Cyber Security - Email Spam Detector")
root.geometry("720x660")
root.configure(bg=current_theme['bg'])

frame = tk.Frame(root, bg=current_theme['bg'], bd=2, relief="groove")
frame.place(relx=0.5, rely=0.5, anchor="center", width=670, height=600)

title = tk.Label(frame, text="🛡️ Cyber Security - Spam Detector", font=("Consolas", 18, "bold"), bg=current_theme['bg'], fg=current_theme['fg'])
title.pack(pady=10)

email_entry = tk.Text(frame, height=7, width=70, font=("Consolas", 12), bd=1, relief="solid", bg=current_theme['entry_bg'], fg=current_theme['fg'])
email_entry.pack(pady=5)

btn_frame = tk.Frame(frame, bg=current_theme['bg'])
btn_frame.pack()

check_button = tk.Button(btn_frame, text="Check Email", command=classify_email, font=("Consolas", 12, "bold"), padx=10, pady=5)
check_button.grid(row=0, column=0, padx=5)

upload_button = tk.Button(btn_frame, text="Upload Email", command=load_email_file, font=("Consolas", 12), padx=10, pady=5)
upload_button.grid(row=0, column=1, padx=5)

save_button = tk.Button(btn_frame, text="Save to CSV", command=save_prediction, font=("Consolas", 12), padx=10, pady=5)
save_button.grid(row=0, column=2, padx=5)

clear_button = tk.Button(btn_frame, text="Clear", command=clear_text, font=("Consolas", 12), padx=10, pady=5)
clear_button.grid(row=0, column=3, padx=5)

theme_button = tk.Button(btn_frame, text="Toggle Theme", command=toggle_theme, font=("Consolas", 12), padx=10, pady=5)
theme_button.grid(row=0, column=4, padx=5)

csv_button = tk.Button(btn_frame, text="📄 Classify CSV", command=process_csv, font=("Consolas", 12), padx=10, pady=5)
csv_button.grid(row=1, column=0, columnspan=5, pady=10)

cleaned_label = tk.Label(frame, text="🔍 Cleaned Email:", font=("Consolas", 12, "bold"), bg=current_theme['bg'], fg=current_theme['fg'])
cleaned_label.pack(pady=5)

cleaned_text_output = tk.Text(frame, height=4, width=70, font=("Consolas", 11), bd=1, relief="solid", bg=current_theme['entry_bg'], fg=current_theme['fg'])
cleaned_text_output.pack(pady=5)

result_label = tk.Label(frame, text="", font=("Consolas", 12, "bold"), bg=current_theme['bg'], fg="gray")
result_label.pack(pady=10)

accuracy_label = tk.Label(frame, text=f"Model Accuracy: {model_accuracy*100:.2f}%", font=("Consolas", 12), bg=current_theme['bg'], fg="gray")
accuracy_label.pack(pady=10)

apply_theme(current_theme)

root.mainloop()
