# Rule-based scoring function
def compute_score(sleep, water, screen_time):
    score = 0
    if sleep >= 7.5:
        score += 2
    elif sleep >= 6:
        score += 1
                          
    if water >= 2.5:
        score += 2
    elif water >= 1.8:
        score += 1
 
    if screen_time <= 3:
        score += 2
    elif screen_time <= 5:
        score += 1
    else:
        score -= 1

    return score

# Generate synthetic dataset (100 samples)
import numpy as np
import pandas as pd

def generate_dataset():
    np.random.seed(42)
    sleep_hours = np.random.uniform(4, 9, 100).round(1)
    water_litres = np.random.uniform(1, 3.5, 100).round(1)
    screen_time = np.random.uniform(1, 8, 100).round(1)

    energy_level = []
    for s, w, st in zip(sleep_hours, water_litres, screen_time):
        score = compute_score(s, w, st)
        if score >= 4:
            energy_level.append("High")
        elif score >= 2:
            energy_level.append("Medium")
        else:
            energy_level.append("Low")

    df = pd.DataFrame({
        "sleep_hours": sleep_hours,
        "water_litres": water_litres,
        "screen_time": screen_time,
        "energy_level": energy_level
    })

    df.to_csv("energy_data_100.csv", index=False)
    print("Dataset saved as energy_data_100.csv")
    print(df.head())

#for model training, trains RandomForest & saves pkls
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def train_model():
    df = pd.read_csv("energy_data_100.csv")

    X = df[["sleep_hours", "water_litres", "screen_time"]]
    le = LabelEncoder()
    y = le.fit_transform(df["energy_level"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))

    pickle.dump(model, open("energy_model.pkl", "wb"))
    pickle.dump(le, open("label_encoder.pkl", "wb"))

    print("Model saved as energy_model.pkl")
    print("Encoder saved as label_encoder.pkl")

# Predict energy level using trained ML model
def predict_energy(sleep, water, screen_time):
    model = pickle.load(open("energy_model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))

    df = pd.DataFrame([[sleep, water, screen_time]],
                      columns=["sleep_hours", "water_litres", "screen_time"])

    pred_idx = model.predict(df)[0]
    label = le.inverse_transform([pred_idx])[0]
    prob = float(model.predict_proba(df).max())

    return label, round(prob, 3)

#rule based prdiction (no ML needed)
def predict_energy_rule(sleep, water, screen_time):
    score = compute_score(sleep, water, screen_time)

    if score >= 4:
        return "High", 0.85
    elif score >= 2:
        return "Medium", 0.70
    else:
        return "Low", 0.45

if __name__ == "__main__":
    print("\n Generating dataset...")
    generate_dataset()

    print("\n Training model...")
    train_model()

    print("\n Testing predictions...")
    print("ML Prediction:", predict_energy(7, 3, 2))
    print("Rule Prediction:", predict_energy_rule(7, 3, 2))
    