import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris

# Load model and dataset
model = joblib.load("model.pkl")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Page config
st.set_page_config(page_title="Iris Classification Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Home page
if page == "Home":
    st.title("🌸 Iris Classification Web App")
    st.markdown("""
    Welcome to the **Iris Classification App** built with **Streamlit** and **Random Forest Classifier**.\n
    Use the sidebar to explore the dataset, view graphs, or try live predictions.
    """)

# Dataset page
elif page == "Dataset":
    st.title("📊 Iris Dataset")
    st.write("Here's a preview of the Iris dataset:")
    st.dataframe(df)

# Summary page
elif page == "Summary":
    st.title("📈 Dataset Summary")
    st.write("Basic statistical description of the dataset:")
    st.write(df.describe())

# Graphs page
elif page == "Graphs":
    st.title("📉 Data Visualization")

    st.subheader("Pairplot of Features")
    fig = sns.pairplot(df, hue='species')
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")

    st.pyplot(plt)

# Predict page
elif page == "Predict":
    st.title("🌼 Iris Flower Prediction")

    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
    sw = st.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
    pw = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)

    if st.button("Predict"):
        data = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(data)[0]
        predicted_species = iris.target_names[prediction]
        st.success(f"🌼 Predicted Species: **{predicted_species}**")
