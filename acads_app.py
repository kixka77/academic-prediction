import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set title
st.title("Academic Performance & Study Habits Predictor")

# Input fields
st.header("Enter Your Study Habits")
study_hours = st.slider("Study hours per week", 0, 100, 10)
sleep_hours = st.slider("Sleep hours per night", 0, 12, 7)
class_attendance = st.slider("Class attendance (%)", 0, 100, 75)
assignments_completed = st.slider("Assignments completed (%)", 0, 100, 80)
extra_activities = st.selectbox("Participates in extra activities?", ["Yes", "No"])

# Button to trigger prediction
if st.button("Predict Academic Performance"):
    # Convert input to numerical
    extra_activities_val = 1 if extra_activities == "Yes" else 0
    user_input = np.array([[study_hours, sleep_hours, class_attendance, assignments_completed, extra_activities_val]])

    # Dummy dataset (replace with real data later)
    np.random.seed(42)
    X = np.random.randint(0, 100, (200, 5))
    y = np.random.choice(["At Risk", "Needs Improvement", "Satisfactory", "Excellent"], 200)

    # Encode labels
    label_map = {"At Risk": 0, "Needs Improvement": 1, "Satisfactory": 2, "Excellent": 3}
    y_num = np.array([label_map[label] for label in y])

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    user_input_scaled = scaler.transform(user_input)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_num, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(user_input_scaled)[0]
    prediction_label = list(label_map.keys())[list(label_map.values()).index(prediction)]

    st.subheader("Prediction Result")
    st.success(f"Your predicted academic performance is: **{prediction_label}**")

    # Feedback
    recommendations = {
        "At Risk": "Seek guidance from mentors, reduce distractions, and create a consistent study schedule.",
        "Needs Improvement": "Try to increase study time and improve consistency with assignments.",
        "Satisfactory": "Maintain current habits, but there's room to optimize study and sleep balance.",
        "Excellent": "Keep up the great work! Continue refining your strategies for even better results."
    }
    st.subheader("Feedback Recommendation")
    st.info(recommendations[prediction_label])

    # Evaluation metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
