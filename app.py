import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ И ТЕМЫ ---
# Устанавливаем широкую верстку и заголовок вкладки
st.set_page_config(page_title="Калькулятор цен: Калифорния", page_icon="🏠", layout="wide")

# Задаем красивую тему для всех графиков Seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rc('figure', figsize=(10, 5)) # Делаем графики по умолчанию шире

# --- ЗАГРУЗКА ДАННЫХ (с кэшированием) ---
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# --- ОБУЧЕНИЕ МОДЕЛИ (с кэшированием) ---
@st.cache_resource
def train_model():
    # Используем меньше деревьев для скорости и стабильности на бесплатном тарифе
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

model = train_model()

# --- ГЛАВНЫЙ ЗАГОЛОВОК И ОПИСАНИЕ ---
st.title("🏠 Умная оценка недвижимости в Калифорнии")
st.markdown("""
    Этот ИИ-ассистент предсказывает стоимость жилья на основе исторических данных. 
    **Настройте параметры дома в панели слева**, чтобы получить мгновенный расчет.
""")
st.divider()

# --- SIDEBAR (Панель управления) ---
# Убираем дублирование, используем только один заголовок
st.sidebar.header("⚙️ Параметры дома")

# Функция для сбора ввода пользователя в правильном порядке
def get_user_input():
    inputs = {}
    
    # Группа 1: Локация
    with st.sidebar.expander("📍 Местоположение", expanded=True):
        lon = st.slider("Долгота", float(X.Longitude.min()), float(X.Longitude.max()), -118.2) # Дефолт: LA
        lat = st.slider("Широта", float(X.Latitude.min()), float(X.Latitude.max()), 34.0)
    
    # Группа 2: Дом
    with st.sidebar.expander("🏗️ Описание здания", expanded=True):
        age = st.slider("Возраст дома (лет)", 1.0, 52.0, 20.0)
        rooms = st.slider("Всего комнат", 1.0, 15.0, 5.0)
        bedrms = st.slider("Из них спален", 1.0, 8.0, 2.0)
        occup = st.slider("Среднее кол-во жильцов", 1.0, 10.0, 3.0)

    # Группа 3: Район
    with st.sidebar.expander("💰 Экономика района"):
        inc = st.slider("Ср. доход населения (в $10k)", 0.5, 15.0, 4.0)
        pop = st.number_input("Население района (чел.)", value=1500, step=100)

    # ОЧЕНЬ ВАЖНО: Соблюдаем точный порядок колонок датасета для модели!
    ordered_input = {
        'MedInc': inc,
        'HouseAge': age,
        'AveRooms': rooms,
        'AveBedrms': bedrms,
        'Population': float(pop),
        'AveOccup': occup,
        'Latitude': lat,
        'Longitude': lon
    }
    return pd.DataFrame(ordered_input, index=[0])

# Получаем данные пользователя
input_df = get_user_input()

# Убедимся, что данные для предсказания не содержат NaN (если Population был пуст)
if input_df.isnull().values.any():
    st.error("Пожалуйста, заполните все поля в панели слева.")
    st.stop()

# --- РАСЧЕТ И ПРЕДСКАЗАНИЕ ---
with st.spinner('ИИ рассчитывает цену...'):
    prediction = model.predict(input_df)[0]
    scaled_price = prediction * 100000 # Цена в датасете в $100k
    avg_price_state = y.mean() * 100000

# --- БЛОК 1: РЕЗУЛЬТАТЫ (Метрики) ---
col1, col2, col3 = st.columns(3)

# Оформляем метрики
col1.metric(
    label="🔮 Предсказанная стоимость", 
    value=f"${scaled_price:,.0f}",
    delta=f"{scaled_price - avg_price_state:,.0f} $ от средней",
    help="Это оценка, основанная на выбранных вами параметрах"
)

col2.metric(
    label="📊 Средняя по Калифорнии", 
    value=f"${avg_price_state:,.0f}",
    help="Средняя цена жилья в исходном датасете"
)

# Вычисляем разницу в процентах
pct_diff = ((scaled_price - avg_price_state) / avg_price_state) * 100
status = "Дороже рынка" if pct_diff > 0 else "Дешевле рынка"
col3.metric(label="⚖️ Статус объекта", value=status, delta=f"{pct_diff:.1f}%")

st.divider()

# --- БЛОК 2: ВИЗУАЛИЗАЦИЯ (Графики) ---
st.header("📊 Анализ стоимости и факторов")

# Две колонки для графиков
g_col1, g_col2 = st.columns([2, 1]) # Левая колонка шире

with g_col1:
    st.subheader("Позиция вашей цены на рынке")
    fig, ax = plt.subplots()
    # Рисуем красивую кривую распределения цен
    sns.histplot(y * 100000, bins=50, kde=True, color="skyblue", ec="white", ax=ax)
    # Добавляем красную вертикальную линию для цены пользователя
    ax.axvline(scaled_price, color='red', linestyle='--', linewidth=2)
    ax.text(scaled_price * 1.05, ax.get_ylim()[1]*0.8, 'Ваш дом', color='red', fontweight='bold')
    
    ax.set_title("Распределение цен на жилье в датасете", fontsize=14)
    ax.set_xlabel("Цена ($)", fontsize=12)
    ax.set_ylabel("Количество районов", fontsize=12)
    # Убираем рамку графиков Seaborn для чистоты
    sns.despine(left=True)
    st.pyplot(fig)

with g_col2:
    st.subheader("🔍 Топ-3 фактора цены")
    # Вычисляем важность признаков
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(3)
    
    fig2, ax2 = plt.subplots()
    # Рисуем горизонтальный барчарт Seaborn
    sns.barplot(x=forest_importances.values, y=forest_importances.index, palette="viridis", ax2=ax2)
    ax2.set_title("Что больше всего влияет?", fontsize=14)
    ax2.set_xlabel("Относительная важность", fontsize=12)
    # Убираем подпись оси Y (там названия факторов)
    ax2.set_ylabel("")
    sns.despine(left=True, bottom=True)
    st.pyplot(fig2)

st.divider()

# --- БЛОК 3: КАРТА И ПОДРОБНОСТИ ---
map_col, text_col = st.columns([2, 1])

with map_col:
    st.header("🗺️ Местоположение на карте")
    # Создаем DataFrame для стандартной карты Streamlit
    # Она работает стабильнее всего и выглядит чисто
    map_data = pd.DataFrame({
        'lat': [input_df.Latitude[0]],
        'lon': [input_df.Longitude[0]]
    })
    st.map(map_data, zoom=6)

with text_col:
    st.header("📋 Ваши параметры")
    # Отображаем таблицу введенных данных вертикально для удобства чтения
    formatted_df = input_df.T
    formatted_df.columns = ["Значение"]
    st.dataframe(formatted_df, use_container_width=True)

# --- ФУТЕР ---
st.divider()
st.caption("Данные: California Housing dataset. Модель: Random Forest Regressor. Разработано с помощью Streamlit.")
