import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="AI Оценка Жилья", page_icon="🏠", layout="wide")

# Кастомный CSS для красоты (закругленные углы и фон)
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1f77b4; }
    .stSlider { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- ЗАГРУЗКА ДАННЫХ И МОДЕЛИ ---
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

# --- БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
with st.sidebar:
    st.title("⚒️ Настройка")
    st.info("Измените параметры ниже, чтобы увидеть, как изменится цена дома.")
    
    with st.expander("📍 Локация", expanded=True):
        lon = st.slider("Долгота", float(X.Longitude.min()), float(X.Longitude.max()), -118.2)
        lat = st.slider("Широта", float(X.Latitude.min()), float(X.Latitude.max()), 34.0)
    
    with st.expander("🏠 Параметры строения", expanded=True):
        age = st.slider("Возраст дома", 1, 52, 25)
        rooms = st.slider("Комнат", 1, 10, 5)
        beds = st.slider("Спален", 1, 5, 2)
        occup = st.slider("Жильцов", 1, 6, 3)
        
    with st.expander("💵 Экономика района"):
        inc = st.slider("Доход ($10k)", 0.5, 15.0, 3.8)
        pop = st.number_input("Население", value=1200)

# Подготовка данных для модели
input_data = pd.DataFrame({
    'MedInc': [inc], 'HouseAge': [float(age)], 'AveRooms': [float(rooms)],
    'AveBedrms': [float(beds)], 'Population': [float(pop)], 'AveOccup': [float(occup)],
    'Latitude': [lat], 'Longitude': [lon]
})

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🏠 Прогноз стоимости недвижимости в Калифорнии")
st.write("Интеллектуальный анализ рыночной стоимости на основе машинного обучения.")

# 1. СЕКЦИЯ МЕТРИК (Главные цифры)
price = model.predict(input_data)[0] * 100000
avg_price = y.mean() * 100000

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Оценочная стоимость", f"${price:,.0f}", delta=f"{price-avg_price:,.0f} $")
with m_col2:
    st.metric("Средняя по штату", f"${avg_price:,.0f}")
with m_col3:
    diff_pct = ((price - avg_price) / avg_price) * 100
    st.metric("Разница с рынком", f"{diff_pct:.1f}%", delta_color="inverse")

st.divider()

# 2. ОСНОВНОЙ КОНТЕНТ (Две колонки)
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📊 Аналитика цен")
    
    # График распределения
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    sns.histplot(y * 100000, bins=40, kde=True, color="#3498db", alpha=0.6)
    plt.axvline(price, color='red', linestyle='--', label='Ваш прогноз')
    plt.title("Где находится ваша цена относительно других?")
    plt.xlabel("Цена дома ($)")
    plt.ylabel("Частота")
    plt.legend()
    st.pyplot(fig)
    
    st.write("**Что это значит?** Красная линия показывает вашу цену на фоне всех домов штата. Если она справа от «горба» — ваш дом дороже большинства.")

with right_col:
    st.subheader("🗺️ География объекта")
    # Маленькая удобная карта
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_df, zoom=7, use_container_width=True)
    
    st.subheader("🔑 Ключевые факторы")
    # Важность признаков (только топ-3)
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(3)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    feat_imp.plot(kind='barh', color='#2ecc71', ax=ax2)
    plt.title("Что больше всего влияет на цену?")
    st.pyplot(fig2)

st.divider()

# 3. ПОДРОБНАЯ ТАБЛИЦА (внизу)
with st.expander("📋 Посмотреть все технические параметры ввода"):
    st.table(input_data)

st.caption("© 2026 AI Real Estate Analyzer. Все расчеты являются вероятностными.")
