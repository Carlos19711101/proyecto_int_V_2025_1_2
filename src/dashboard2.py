import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

def calcular_indicadores(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = ma20 + 2*std20
    df['Bollinger_Lower'] = ma20 - 2*std20
    
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(window=14).mean()
    
    df['Media Móvil 30'] = df['Close'].rolling(window=30).mean()
    
    return df

@st.cache_data
def load_data():
    path = Path('src/XRP/static/data/procesados.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def main():
    st.title("Dashboard Interactivo de KPIs Cripto")

    df = load_data()
    df = calcular_indicadores(df)

    fecha_min = df.index.min().date()
    fecha_max = df.index.max().date()
    fecha_inicio = st.date_input("Fecha inicio", fecha_min, min_value=fecha_min, max_value=fecha_max)
    fecha_fin = st.date_input("Fecha fin", fecha_max, min_value=fecha_min, max_value=fecha_max)

    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha fin.")
        return

    df_filtrado = df.loc[fecha_inicio:fecha_fin]

    # Filtrar últimos 60 días para gráficos y KPIs
    fecha_max_60 = df_filtrado.index.max()
    fecha_min_60 = fecha_max_60 - pd.Timedelta(days=60)
    df_ultimos_60 = df_filtrado.loc[fecha_min_60:fecha_max_60]

    # KPIs para mostrar valores actuales (últimos 60 días)
    kpis = {
        "Tasa de Variación (%)": df_ultimos_60['Close'].pct_change(),
        "Media Móvil 30": df_ultimos_60['Media Móvil 30'],
        "Volatilidad 30": df_ultimos_60['Close'].pct_change().rolling(window=30).std() * np.sqrt(30),
        "Retorno Acumulado": (1 + df_ultimos_60['Close'].pct_change()).cumprod() - 1,
        "Desviación Estándar 30": df_ultimos_60['Close'].rolling(window=30).std(),
        "RSI": df_ultimos_60['RSI'],
        "MACD": df_ultimos_60['MACD'],
        "ADX": df_ultimos_60['ADX']
    }

    # Mostrar valores actuales de KPIs
    cols = st.columns(len(kpis))
    for i, (label, serie) in enumerate(kpis.items()):
        valor = serie.iloc[-1]
        if pd.isna(valor):
            display_val = "N/A"
        else:
            display_val = f"{valor:.4f}" if abs(valor) < 100 else f"{valor:.2f}"
        cols[i].metric(label, display_val)

    st.markdown("---")

    # Mostrar gráfico de barras para cada KPI (últimos 60 días)
    for label, serie in kpis.items():
        st.subheader(f"Evolución de {label} (últimos 60 días)")
        # Preparar dataframe para Plotly (convertir índice a string para eje x)
        df_plot = pd.DataFrame({
            'Fecha': serie.index.strftime('%Y-%m-%d'),
            label: serie.values
        }).dropna()

        fig = px.bar(df_plot, x='Fecha', y=label, labels={'Fecha': 'Fecha', label: label},
                     title=f'Gráfico de barras de {label}', height=400)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
# Para correr el dashboard:
# streamlit run src/dashboard2.py   