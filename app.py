import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- ИНИЦИАЛИЗАЦИЯ ---
st.set_page_config(page_title="AI Оценка Недвижимости", page_icon="🏠", layout="wide")

# Кастомный стиль для блоков
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stPlotlyChart { border-radius: 10px; }
    div[data-testid="stMetric"] {
        background-color: #1e2129;
        border: 1px solid #31333f;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

X, y = load_data()
model = train_model(X, y)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Настройки объекта")
    
    with st.expander("📍 География", expanded=True):
        lon = st.slider("Долгота", float(X.Longitude.min()), float(X.Longitude.max()), -118.2)
        lat = st.slider("Широта", float(X.Latitude.min()), float(X.Latitude.max()), 34.0)
    
    with st.expander("🏗️ Строение", expanded=True):
        age = st.slider("Возраст дома", 1, 52, 25)
        rooms = st.slider("Всего комнат", 1, 10, 5)
        beds = st.slider("Всего спален", 1, 6, 2)
        occup = st.slider("Жильцов (ср)", 1, 6, 3)
        
    with st.expander("💰 Социо-экономика"):
        inc = st.slider("Доход района ($10k)", 0.5, 15.0, 3.8)
        pop = st.number_input("Население (чел)", value=1200)

input_data = pd.DataFrame({
    'MedInc': [inc], 'HouseAge': [float(age)], 'AveRooms': [float(rooms)],
    'AveBedrms': [float(beds)], 'Population': [float(pop)], 'AveOccup': [float(occup)],
    'Latitude': [lat], 'Longitude': [lon]
})

# --- ОСНОВНОЙ ЭКРАН ---
st.title("🏠 Система AI-аналитики недвижимости")
st.caption("Профессиональный инструмент оценки стоимости жилья в штате Калифорния")

# 1. СЕКЦИЯ МЕТРИК
price = model.predict(input_data)[0] * 100000
avg_price = y.mean() * 100000
diff_pct = ((price - avg_price) / avg_price) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Прогноз цены", f"${price:,.0f}")
m2.metric("Средняя по штату", f"${avg_price:,.0f}")
m3.metric("Разница $", f"{price - avg_price:+,.0f}$")
m4.metric("Разница %", f"{diff_pct:+.1f}%")

st.write("---")

# 2. ОСНОВНОЙ БЛОК АНАЛИЗА (ДВЕ КОЛОНКИ С ОДИНАКОВОЙ ВЫСОТОЙ)
col_main_left, col_main_right = st.columns([1, 1])

with col_main_left:
    st.subheader("📊 Анализ рыночной позиции")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.histplot(y * 100000, bins=50, kde=True, color="#00d4ff", alpha=0.4)
    plt.axvline(price, color='red', linestyle='--', linewidth=3, label='Ваш прогноз')
    plt.title("Распределение цен в Калифорнии", fontsize=15, pad=20)
    plt.xlabel("Стоимость дома ($)")
    plt.legend()
    st.pyplot(fig)
    
    with st.expander("💡 Что это значит?"):
        st.write(f"""
        Ваш объект оценивается в **${price:,.0f}**. 
        Это значение находится в {'верхнем' if price > avg_price else 'нижнем'} ценовом сегменте штата. 
        Основное влияние на эту цифру оказал параметр **{X.columns[np.argmax(model.feature_importances_)]}**.
        """)

with col_main_right:
    st.subheader("🗺️ Точное расположение")
    # Карта теперь вровень с графиком
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_df, zoom=8, use_container_width=True)
    
    st.subheader("🔑 Влияние факторов")
    # Горизонтальный график важности
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    colors = ['#2c3e50' if x < feat_imp.max() else '#e74c3c' for x in feat_imp]
    feat_imp.plot(kind='barh', color=colors, ax=ax2)
    plt.title("Вес параметров в итоговой цене", fontsize=12)
    st.pyplot(fig2)

st.write("---")

# 3. НИЖНЯЯ СЕКЦИЯ: СРАВНЕНИЕ И ДЕТАЛИ
st.header("🔍 Детальное сравнение")
down_col1, down_col2 = st.columns(2)

with down_col1:
    st.write("**Ваши показатели vs Средние по штату**")
    comparison = pd.DataFrame({
        "Ваш выбор": input_data.iloc[0],
        "Среднее": X.mean()
    })
    # Нормализуем для графика (просто для наглядности)
    st.bar_chart(comparison)

with down_col2:
    st.write("**Справочник параметров**")
    descriptions = {
        "MedInc": "Средний доход семей в районе (влияет сильнее всего)",
        "HouseAge": "Средний возраст зданий в округе",
        "AveRooms": "Среднее количество комнат в домах",
        "AveOccup": "Среднее количество жильцов в одном доме",
        "Population": "Общее количество людей в этом районе"
    }
    for k, v in descriptions.items():
        st.write(f"**{k}**: {v}")

st.divider()
st.caption("Данные актуальны на основе California Housing Dataset. Разработка: AI Predictor Lab.")
