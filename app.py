import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Ensemble Learning", page_icon="🤖")

st.title("🤖 Ensemble Learning Prediction App")
st.write("Random Forest Regression Demo")

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 45, 50, 55, 62, 68, 75, 82, 90]
}

df = pd.DataFrame(data)

st.subheader("📊 Student Dataset")
st.dataframe(df)

# Input and output
X = df[["Hours_Studied", "Attendance"]]
y = df["Marks"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("🎯 Predict Student Marks")

hours = st.slider("Hours Studied", 1, 10, 5)
attendance = st.slider("Attendance (%)", 50, 100, 75)

if st.button("Predict Marks"):
    prediction = model.predict([[hours, attendance]])[0]

    st.success(f"Predicted Marks: {prediction:.2f}")

    # Graph
    fig, ax = plt.subplots()
    ax.plot(df["Hours_Studied"], df["Marks"], marker="o")
    ax.scatter(hours, prediction, s=100)
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Marks")
    ax.set_title("Hours Studied vs Marks")

    st.pyplot(fig)

# Pie chart
st.subheader("🥧 Dataset Distribution")

fig2, ax2 = plt.subplots()
ax2.pie(
    [30, 40, 30],
    labels=["Low Marks", "Average Marks", "High Marks"],
    autopct="%1.1f%%"
)

st.pyplot(fig2)
