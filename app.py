import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="California Home AI", page_icon="🏠", layout="wide")

# Стиль для кнопок и метрик
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# --- ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- ЗАГОЛОВОК ---
st.title("🏠 California Real Estate AI Predictor")
st.write("Используйте панель слева, чтобы настроить параметры дома и получить мгновенную оценку стоимости.")

# --- SIDEBAR (ВВОД ДАННЫХ) ---
st.sidebar.header("⚙️ Характеристики дома")

def user_input_features():
    inputs = {}
    # Группируем слайдеры по смыслу
    with st.sidebar.expander("📍 Локация и возраст", expanded=True):
        inputs['Longitude'] = st.slider("Долгота", float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))
        inputs['Latitude'] = st.slider("Широта", float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
        inputs['HouseAge'] = st.slider("Возраст дома (лет)", 1, 52, 28)

    with st.sidebar.expander("🏗️ Параметры здания", expanded=True):
        inputs['AveRooms'] = st.slider("Среднее кол-во комнат", 1.0, 10.0, 5.0)
        inputs['AveBedrms'] = st.slider("Среднее кол-во спален", 1.0, 5.0, 1.0)
        inputs['AveOccup'] = st.slider("Жильцов в доме (среднее)", 1.0, 6.0, 3.0)

    with st.sidebar.expander("💰 Экономика района"):
        inputs['MedInc'] = st.slider("Средний доход (в $10k)", 0.5, 15.0, 3.8)
        inputs['Population'] = st.number_input("Население района", value=1500)

    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# --- РАСЧЕТ ПРЕДСКАЗАНИЯ ---
prediction = model.predict(input_df)[0]
scaled_price = prediction * 100000
avg_price = y.mean() * 100000

# --- ГЛАВНЫЕ МЕТРИКИ ---
c1, c2, c3 = st.columns(3)
delta = scaled_price - avg_price

with c1:
    st.metric("Предсказанная цена", f"${scaled_price:,.0f}", delta=f"{delta:,.0f}")
with c2:
    st.metric("Средняя по штату", f"${avg_price:,.0f}")
with c3:
    status = "Выше рынка" if delta > 0 else "Ниже рынка"
    st.metric("Статус объекта", status)

st.divider()

# --- ВКЛАДКИ ---
tab1, tab2, tab3 = st.tabs(["📊 Аналитика", "🗺️ Интерактивная карта", "🔍 Важность факторов"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Распределение цен в Калифорнии")
        fig_hist = px.histogram(y*100000, nbins=50, title="Где находится ваша цена?", 
                               labels={'value': 'Цена ($)'}, color_discrete_sequence=['#636EFA'])
        fig_hist.add_vline(x=scaled_price, line_dash="dash", line_color="red", annotation_text="Ваш выбор")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Сравнение со средним")
        comparison_df = pd.DataFrame({
            "Параметр": X.columns,
            "Ваш дом": input_df.iloc[0].values,
            "Среднее по штату": X.mean().values
        }).melt(id_vars="Параметр", var_name="Тип", value_name="Значение")
        
        fig_comp = px.bar(comparison_df, x="Параметр", y="Значение", color="Тип", barmode="group",
                          title="Ваши параметры vs Средние")
        st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    st.subheader("Географический анализ цен")
    # Создаем красивую карту через Plotly
    map_df = X.sample(2000).copy() # берем выборку для скорости
    map_df['Price'] = y.sample(2000).values * 100000
    
    fig_map = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", color="Price", size="Price",
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5,
                                mapbox_style="carto-positron", title="Цены на жилье по координатам")
    
    # Добавляем точку пользователя на карту
    fig_map.add_scattermapbox(lat=input_df["Latitude"], lon=input_df["Longitude"], 
                              marker=dict(size=20, color='red'), name="Ваш объект")
    
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Что больше всего влияет на цену?")
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                     title="Важность признаков по версии AI")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.info("💡 Как видно из графика, доход населения (MedInc) — ключевой фактор стоимости.")

st.sidebar.success("Готово! Все данные обновлены.")
