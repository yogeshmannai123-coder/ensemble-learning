import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Student Exam Score Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Exam Score Prediction")
st.subheader("Using Educational Data Mining and Machine Learning")

st.write(
    "This application predicts a student's exam score "
    "using educational data such as study hours, attendance, "
    "assignment score and previous marks."
)


# --------------------------------------------------
# CREATE EDUCATIONAL DATA
# --------------------------------------------------

np.random.seed(42)

study_hours = np.random.randint(1, 11, 100)
attendance = np.random.randint(50, 101, 100)
assignment_score = np.random.randint(40, 101, 100)
previous_marks = np.random.randint(40, 101, 100)

exam_score = (
    0.30 * study_hours * 10
    + 0.25 * attendance
    + 0.20 * assignment_score
    + 0.25 * previous_marks
    + np.random.normal(0, 5, 100)
)

exam_score = np.clip(exam_score, 0, 100)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Previous_Marks": previous_marks,
    "Exam_Score": exam_score
})


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📚 Student Information")

student_name = st.sidebar.text_input(
    "Student Name",
    "Student"
)

study = st.sidebar.slider(
    "Study Hours per Day",
    1,
    10,
    5
)

attend = st.sidebar.slider(
    "Attendance (%)",
    50,
    100,
    75
)

assignment = st.sidebar.slider(
    "Assignment Score",
    40,
    100,
    75
)

previous = st.sidebar.slider(
    "Previous Exam Marks",
    40,
    100,
    70
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

st.header("📊 Educational Dataset")

st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# DATA INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Students", len(df))

with col2:
    st.metric("Average Exam Score", round(df["Exam_Score"].mean(), 2))

with col3:
    st.metric("Average Attendance", round(df["Attendance"].mean(), 2))


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

st.header("📈 Data Visualization")

fig, ax = plt.subplots()

ax.scatter(
    df["Study_Hours"],
    df["Exam_Score"]
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")
ax.set_title("Study Hours vs Exam Score")

st.pyplot(fig)


# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Marks"
    ]
]

y = df["Exam_Score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(
    y_test,
    prediction
)

r2 = r2_score(
    y_test,
    prediction
)


st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        round(mae, 2)
    )

with col2:
    st.metric(
        "MSE",
        round(mse, 2)
    )

with col3:
    st.metric(
        "R² Score",
        round(r2, 2)
    )


# --------------------------------------------------
# HYPERPARAMETER TUNING
# --------------------------------------------------

st.header("⚙️ Hyperparameter Tuning")

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(
        random_state=42
    ),
    param_grid,
    cv=3,
    scoring="r2"
)

grid_search.fit(
    X_train,
    y_train
)

st.write(
    "Best Parameters:"
)

st.json(
    grid_search.best_params_
)


# --------------------------------------------------
# STUDENT SCORE PREDICTION
# --------------------------------------------------

st.header("🎯 Predict Student Exam Score")

if st.button("Predict Exam Score"):

    input_data = pd.DataFrame({
        "Study_Hours": [study],
        "Attendance": [attend],
        "Assignment_Score": [assignment],
        "Previous_Marks": [previous]
    })

    best_model = grid_search.best_estimator_

    predicted_score = best_model.predict(
        input_data
    )[0]

    predicted_score = np.clip(
        predicted_score,
        0,
        100
    )

    st.success(
        f"🎓 {student_name}'s Predicted Exam Score: "
        f"{predicted_score:.2f} / 100"
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.header("🔍 Feature Importance")

importance = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)


# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------

st.header("🔄 Educational Data Mining Workflow")

st.write("""
**Student Data**
↓
**Data Collection**
↓
**Data Cleaning**
↓
**Data Visualization**
↓
**Feature Selection**
↓
**Train/Test Split**
↓
**Random Forest**
↓
**Hyperparameter Tuning**
↓
**Prediction**
↓
**Model Evaluation**
""")

st.info(
    "This project demonstrates how Educational Data Mining "
    "and Machine Learning can be used to predict student exam performance."
)
