import streamlit as st
from sklearn.datasets import fetch_california_housing
import pandas as pd

st.title('My first project')
st.markdown("### Smart real estate valuation powered by Machine Learning")

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()
