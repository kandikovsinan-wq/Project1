import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- ИНИЦИАЛИЗАЦИЯ И СТИЛЬ ---
st.set_page_config(page_title="California AI Expert", page_icon="💎", layout="wide")

# Кастомный темный дизайн с золотыми акцентами
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border-left: 5px solid #ffd700;
        padding: 20px;
    }
    .verdict-box {
        background-color: #262730;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #444;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df, housing.feature_names

df, features = load_data()

@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df[features], df['PRICE'])
    return model

model = train_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏯 Параметры")
    st.write("Настройте характеристики для анализа")
    
    inc = st.slider("💰 Доход района ($10k)", 0.5, 15.0, 5.0)
    age = st.slider("⏳ Возраст дома", 1, 52, 20)
    rooms = st.slider("🛏️ Комнат", 1, 10, 4)
    lon = st.slider("📍 Долгота", float(df.Longitude.min()), float(df.Longitude.max()), -118.2)
    lat = st.slider("📍 Широта", float(df.Latitude.min()), float(df.Latitude.max()), 34.0)
    pop = st.number_input("👥 Население", value=1000)
    occup = st.slider("👨‍👩‍👧 Жильцов (ср)", 1.0, 6.0, 3.0)
    beds = st.slider("🚿 Спален (ср)", 1.0, 5.0, 1.0)

# Подготовка данных
user_input = pd.DataFrame({
    'MedInc': [inc], 'HouseAge': [float(age)], 'AveRooms': [float(rooms)],
    'AveBedrms': [float(beds)], 'Population': [float(pop)], 'AveOccup': [float(occup)],
    'Latitude': [lat], 'Longitude': [lon]
})

# --- ГЛАВНЫЙ ЭКРАН ---
st.title("💎 California Real Estate AI Expert")
st.markdown("---")

# 1. СЕКЦИЯ: ГЛАВНЫЙ ВЕРДИКТ (Уникальная фишка)
price = model.predict(user_input)[0] * 100000
avg_p = df['PRICE'].mean() * 100000
diff = ((price - avg_p) / avg_p) * 100

st.subheader("🤖 Заключение нейросети")
verdict_color = "#ff4b4b" if diff > 15 else "#28a745" if diff < -15 else "#ffd700"

# Логика уникального вердикта
if diff > 30:
    v_text = "⚠️ **Элитный сегмент.** Цена значительно выше средней. Вероятно, это эксклюзивный район или премиальное жилье."
elif diff < -30:
    v_text = "🔥 **Выгодная сделка!** Цена аномально низкая для Калифорнии. Идеально для инвестиций или первой покупки."
else:
    v_text = "✅ **Рыночная стабильность.** Цена соответствует средним показателям. Надежный вариант без переплат."

st.markdown(f"""
    <div class="verdict-box">
        <h2 style='color: {verdict_color}; margin-top:0;'>${price:,.0f}</h2>
        <p style='font-size: 1.2rem;'>{v_text}</p>
    </div>
    """, unsafe_allow_html=True)

# 2. СЕКЦИЯ: ИНДЕКС ПРИВЛЕКАТЕЛЬНОСТИ
st.write("---")
st.header("📈 Аналитический отчет")

col1, col2 = st.columns([1.5, 1])

with col1:
    # График распределения с акцентом
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.set_style("dark")
    sns.kdeplot(df['PRICE'] * 100000, fill=True, color="#ffd700", alpha=0.2, ax=ax1)
    plt.axvline(price, color=verdict_color, linestyle='--', linewidth=3, label="Ваш объект")
    plt.title("Позиция на рынке (Цена vs Плотность предложений)", color="white")
    ax1.tick_params(colors='white')
    st.pyplot(fig1)

with col2:
    # Инвестиционный потенциал (уникальный расчет)
    st.write("**Инвестиционный потенциал**")
    score = 100 - abs(diff) # Чем ближе к средней, тем "безопаснее"
    st.progress(min(max(int(score), 0), 100))
    st.caption("Шкала показывает баланс между ценой и рыночным риском.")

# 3. СЕКЦИЯ: ГЕО-ВИЗУАЛИЗАЦИЯ
st.write("---")
st.header("🗺️ Локация и окружение")
st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=9)

# 4. СЕКЦИЯ: ГЛУБОКИЕ ДАННЫЕ
with st.expander("🔍 Посмотреть детальную матрицу влияния факторов"):
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    feat_imp = pd.Series(model.feature_importances_, index=features).sort_values()
    feat_imp.plot(kind='barh', color="#ffd700")
    plt.title("Почему ИИ выбрал такую цену?", color="white")
    st.pyplot(fig2)

st.write("---")
st.caption("Разработано с использованием Random Forest Regressor | California Housing Data")
