import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Title
st.title("Academic Performance Predictor with Feedback")

# Collect user input
study_hours = st.slider("Study Hours per Week", 0, 100, 25)
sleep_hours = st.slider("Average Sleep Hours per Day", 0, 12, 6)
attendance = st.slider("Attendance Percentage", 0, 100, 75)
assignments = st.slider("Assignment Completion Rate (%)", 0, 100, 80)
activities = st.selectbox("Are you active in extracurricular activities?", ["Yes", "No"])
activities = 1 if activities == "Yes" else 0

user_input = np.array([[study_hours, sleep_hours, attendance, assignments, activities]])

# ----- Generate Synthetic Data -----
data = []
labels = []
for _ in range(300):
    sh = np.random.randint(0, 100)
    sl = np.random.randint(0, 12)
    att = np.random.randint(0, 100)
    assign = np.random.randint(0, 100)
    act = np.random.choice([0, 1])
    
    score = sh * 0.3 + sl * 0.2 + att * 0.2 + assign * 0.2 + act * 10
    if score < 50:
        label = "At Risk"
    elif score < 65:
        label = "Needs Improvement"
    elif score < 80:
        label = "Satisfactory"
    else:
        label = "Excellent"
    
    data.append([sh, sl, att, assign, act])
    labels.append(label)

X = np.array(data)
y = np.array(labels)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Predict Button
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)
    predicted_label = le.inverse_transform(prediction)[0]

    st.subheader("Prediction:")
    st.success(f"Your predicted academic performance category is: **{predicted_label}**")

    st.subheader("Evaluation Metrics:")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

    st.subheader("Feedback & Recommendations:")
    if predicted_label == "At Risk":
        st.warning("You're at risk academically. Try increasing your study time and reducing distractions.")
    elif predicted_label == "Needs Improvement":
        st.info("You’re close! Improve your sleep and assignment submission rates for better results.")
    elif predicted_label == "Satisfactory":
        st.success("You’re doing okay. Stay consistent and aim higher!")
    elif predicted_label == "Excellent":
        st.balloons()
        st.success("Excellent! Keep up the great performance!")
