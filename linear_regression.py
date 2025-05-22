import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('cleaned_data.csv')

# Prepare input and output variables
x = df[['CGPA']]
y = df[['Package(LPA)']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Streamlit app interface
st.title("🎓 CGPA vs Placement Package Predictor 💼")

st.markdown("This app uses Linear Regression to predict the package (in LPA) based on CGPA.")

# Sidebar input
cgpa_input = st.number_input("Enter CGPA:", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    if cgpa_input < 6.0:
        st.error("❌ You are not eligible for placement (Minimum CGPA required: 6.0)")
    else:
        prediction = model.predict([[cgpa_input]])[0][0]
        st.success(f"Predicted Package for CGPA {cgpa_input} is: ₹ {prediction:.2f} LPA")
        st.metric("Model Accuracy (R² Score)", f"{model.score(x_test, y_test) * 100:.2f}%")
        # Plotting
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='CGPA', y='Package(LPA)', data=df, ax=ax)
        ax.plot(x, model.predict(x), color='red')
        ax.set_title('CGPA vs Package')
        ax.set_xlabel('CGPA')
        ax.set_ylabel('Package (LPA)')
        ax.grid(True)
        st.pyplot(fig)
