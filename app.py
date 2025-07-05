# Save this as app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6],
    'Attendance': [60, 70, 75, 80, 90, 95],
    'InternalMarks': [10, 15, 20, 22, 25, 28],
    'FinalMarks': [35, 45, 50, 60, 70, 75]
}
df = pd.DataFrame(data)

# Train the model
X = df[['StudyHours', 'Attendance', 'InternalMarks']]
y = df['FinalMarks']
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Marks Predictor")
study_hours = st.slider("📚 Study Hours", 1, 10)
attendance = st.slider("📅 Attendance (%)", 50, 100)
internal_marks = st.slider("📝 Internal Marks", 0, 30)

if st.button("Predict Final Marks"):
    prediction = model.predict([[study_hours, attendance, internal_marks]])
    st.success(f"🎯 Predicted Marks: {round(prediction[0], 2)}")
