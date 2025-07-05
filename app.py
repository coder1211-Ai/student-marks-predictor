import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 🎓 Title
st.title("🎓 Student Final Marks Predictor (ML Powered)")

# 📂 Load CSV file
df = pd.read_csv("student_scores.csv")  # Make sure this CSV is in your repo

# 🧾 Show sample data
st.subheader("📄 Sample Training Data")
st.write(df.head())

# 🔍 Split into input features (X) and target label (y)
X = df[['StudyHours', 'Attendance', 'InternalMarks']]
y = df['FinalMarks']

# 🔧 Train the ML model
model = LinearRegression()
model.fit(X, y)

# 📊 User Input
st.subheader("📥 Enter Student Details")
study_hours = st.slider("📚 Study Hours", 1, 10)
attendance = st.slider("📅 Attendance (%)", 50, 100)
internal_marks = st.slider("📝 Internal Marks", 0, 30)

# 🎯 Predict button
if st.button("Predict Final Marks"):
    prediction = model.predict([[study_hours, attendance, internal_marks]])
    marks = round(prediction[0], 2)

    st.success(f"🎯 Predicted Final Marks: {marks} / 100")

    if marks >= 40:
        st.success("✅ Status: Pass")
    else:
        st.error("❌ Status: Fail")

# 📈 Optional Visualization
st.subheader("📈 Study Hours vs Final Marks Chart")
plt.scatter(df['StudyHours'], df['FinalMarks'], color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.grid(True)
st.pyplot(plt)
