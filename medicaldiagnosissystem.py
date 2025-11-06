import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def create_synthetic_data(csv_path="sample_symptoms.csv", n_rows=400, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    symptoms = [
        "fever", "cough", "sore_throat", "fatigue", "headache",
        "nausea", "vomiting", "diarrhea", "shortness_of_breath",
        "chest_pain", "runny_nose", "muscle_ache", "loss_of_taste",
        "loss_of_smell", "rash"
    ]

    diseases = [
        "Common Cold", "Flu", "COVID-19", "Gastroenteritis",
        "Migraine", "Food Poisoning", "Allergy", "Pneumonia"
    ]

    df_rows = []
    for _ in range(n_rows):
        # Choose a base disease
        disease = random.choice(diseases)

        # Base symptom pattern by disease (very rough and synthetic)
        row = {sym: 0 for sym in symptoms}
        if disease == "Common Cold":
            picks = ["cough", "runny_nose", "sore_throat", "headache"]
        elif disease == "Flu":
            picks = ["fever", "cough", "fatigue", "muscle_ache", "headache"]
        elif disease == "COVID-19":
            picks = ["fever", "cough", "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath"]
        elif disease == "Gastroenteritis":
            picks = ["nausea", "vomiting", "diarrhea", "fatigue"]
        elif disease == "Migraine":
            picks = ["headache", "nausea", "vomiting", "fatigue"]
        elif disease == "Food Poisoning":
            picks = ["nausea", "vomiting", "diarrhea", "fever"]
        elif disease == "Allergy":
            picks = ["runny_nose", "rash", "cough", "sore_throat"]
        else:  # Pneumonia
            picks = ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"]

        # Turn some picks on, plus a few random noises
        for s in picks:
            row[s] = 1

        # Random noise symptoms
        noise = random.sample(symptoms, k=random.randint(0, 2))
        for s in noise:
            row[s] = 1 if random.random() < 0.3 else row[s]

        row["disease"] = disease
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df.to_csv(csv_path, index=False)
    print("Saved synthetic dataset to:")
    print(csv_path)
    print("Head:")
    print(df.head())
    return df

def train_and_evaluate(df):
    y = df["disease"]
    X = df.drop(columns=["disease"])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:")
    print(acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    return model, le, list(X.columns)

def cli_predict(model, label_encoder, symptom_cols):
    print("")
    print("Simple CLI prediction demo")
    print("Type comma-separated symptoms from this list:")
    print(", ".join(symptom_cols))
    print("Example: fever, cough, fatigue")
    try:
        user_in = input("Enter symptoms (or just press Enter to skip): ").strip()
    except EOFError:
        user_in = ""
    if not user_in:
        print("No input given. Skipping prediction.")
        return
    typed = [s.strip().lower().replace(" ", "_") for s in user_in.split(",") if s.strip()]

    # Build input vector
    x = []
    for s in symptom_cols:
        x.append(1 if s in typed else 0)
    X = np.array(x).reshape(1, -1)

    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    print("Predicted disease:")
    print(pred_label)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        labels = list(label_encoder.classes_)
        pairs = list(zip(labels, probs))
        pairs.sort(key=lambda t: t[1], reverse=True)
        top3 = pairs[:3]
        print("Top probabilities:")
        for d, p in top3:
            print(d + ": " + str(round(float(p), 3)))

def main():
    # 1) Create data
    csv_path = "sample_symptoms.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("Found existing dataset:")
        print(csv_path)
        print("Head:")
        print(df.head())
    else:
        df = create_synthetic_data(csv_path=csv_path, n_rows=400, seed=42)

    # 2) Train & evaluate
    model, le, symptom_cols = train_and_evaluate(df)

    # 3) CLI predict
    cli_predict(model, le, symptom_cols)

    print("")
    print("Done. Files you can use:")
    print("Data CSV: sample_symptoms.csv")
    print("Note: This is an educational demo; not medical advice.")

if __name__ == "__main__":
    main()
