import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("data/student_performance.csv")
    features = [
        "study_hours",
        "attendance",
        "assignment_score",
        "previous_score",
        "sleep_hours",
        "extracurricular_hours",
    ]
    X = df[features]
    y = df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, df

model, accuracy, data = train_model()

st.title("🎓 AI Student Performance Predictor")
st.write(
    "Enter a student's academic and study-related information. "
    "The machine-learning model predicts whether the student is "
    "**At Risk, Average, or Excellent**."
)

with st.form("prediction_form"):
    study_hours = st.slider("Daily study hours", 0.5, 8.0, 4.0, 0.1)
    attendance = st.slider("Attendance (%)", 55.0, 100.0, 80.0, 0.5)
    assignment_score = st.slider("Assignment score (%)", 40.0, 100.0, 75.0, 0.5)
    previous_score = st.slider("Previous exam score (%)", 35.0, 95.0, 70.0, 0.5)
    sleep_hours = st.slider("Average sleep per day", 4.5, 9.0, 7.0, 0.1)
    extracurricular_hours = st.slider("Extracurricular hours per day", 0.0, 5.0, 1.5, 0.1)
    submitted = st.form_submit_button("Predict Performance")

if submitted:
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_score": assignment_score,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "extracurricular_hours": extracurricular_hours,
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    confidence = probabilities[list(classes).index(prediction)] * 100

    st.subheader(f"Prediction: {prediction}")
    st.metric("Model confidence", f"{confidence:.1f}%")

    if prediction == "At Risk":
        st.warning(
            "Recommendation: increase study time, maintain attendance, "
            "complete assignments consistently, and review weak topics."
        )
    elif prediction == "Average":
        st.info(
            "Recommendation: maintain the current routine and focus on "
            "improving study consistency and previous weak areas."
        )
    else:
        st.success(
            "Recommendation: continue the current routine and challenge "
            "yourself with advanced topics or projects."
        )

st.divider()
st.caption(f"Test-set accuracy: {accuracy * 100:.1f}% | Training records: {len(data)}")
st.caption("This is an educational project using a sample dataset; it is not a substitute for academic evaluation.")
