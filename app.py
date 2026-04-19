import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

# Настройка страницы
st.set_page_config(page_title="Mall Segments", layout="wide")

st.title("🛍️ Сегментация покупателей")
st.markdown("Переключайте параметры в боковой панели, чтобы увидеть, как меняются кластеры.")

# 1. Загрузка данных (с кешированием, чтобы не тормозило)
@st.cache_data
def get_data():
    df = pd.read_csv('Mall_Customers.csv')
    X_raw = df.drop(columns=['CustomerID'])
    X_raw['Gender'] = X_raw['Gender'].map({'Male': 0, 'Female': 1})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_raw, X_scaled

data_raw, X_scaled = get_data()

# 2. Боковая панель (Sidebar)
st.sidebar.header("Настройки")
k_val = st.sidebar.slider("Число кластеров (K)", 2, 10, 5)
method = st.sidebar.radio("Метод проекции", ["PCA", "UMAP"])

# 3. Кластеризация в реальном времени
kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 4. Снижение размерности
if method == "PCA":
    reducer = PCA(n_components=2)
else:
    # Для UMAP в приложении лучше зафиксировать random_state для стабильности картинки
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

X_2d = reducer.fit_transform(X_scaled)

# 5. Создание интерактивного графика Plotly
viz_df = pd.DataFrame(X_2d, columns=['x', 'y'])
viz_df['Cluster'] = labels.astype(str)
# Добавляем данные для красивых подсказок
viz_df['Income'] = data_raw['Annual Income (k$)'].values
viz_df['Score'] = data_raw['Spending Score (1-100)'].values

fig = px.scatter(
    viz_df, x='x', y='y', color='Cluster',
    title=f"Визуализация через {method}",
    hover_data=['Income', 'Score'],
    template="plotly_white"
)

# Вывод графика
st.plotly_chart(fig, use_container_width=True)
