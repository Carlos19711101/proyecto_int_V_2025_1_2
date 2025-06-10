import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

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

def mostrar_documentacion():
    st.sidebar.header("Documentación de KPIs")
    st.sidebar.markdown("""
    **Tasa de Variación (%)**: Muestra el cambio porcentual diario del precio de cierre. Indica la volatilidad diaria.

    **Media Móvil 30**: Promedio del precio de cierre en los últimos 30 días. Ayuda a suavizar fluctuaciones y detectar tendencias.

    **Volatilidad 30**: Desviación estándar anualizada del cambio porcentual diario en 30 días. Mide la incertidumbre o riesgo.

    **Retorno Acumulado (%)**: Ganancia o pérdida total acumulada en el periodo seleccionado.

    **Desviación Estándar 30**: Medida de dispersión del precio de cierre en 30 días.

    **RSI (Índice de Fuerza Relativa)**: Oscilador que indica condiciones de sobrecompra (>70) o sobreventa (<30).

    **MACD (Convergencia/Divergencia de Medias Móviles)**: Indica cambios en la fuerza, dirección y duración de una tendencia.

    **ADX (Índice Direccional Promedio)**: Mide la fuerza de la tendencia sin importar dirección. Valores >25 indican tendencia fuerte.
    """)

def main():
    st.title("Dashboard para Presentación a Inversionistas")

    mostrar_documentacion()

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

    # Últimos 60 días para KPIs y gráficos
    fecha_max_60 = df_filtrado.index.max()
    fecha_min_60 = fecha_max_60 - pd.Timedelta(days=60)
    df_ultimos_60 = df_filtrado.loc[fecha_min_60:fecha_max_60]

    # KPIs actuales para mostrar métricas
    kpis = {
        "Tasa de Variación (%)": df_ultimos_60['Close'].pct_change() * 100,
        "Media Móvil 30": df_ultimos_60['Media Móvil 30'],
        "Volatilidad 30": df_ultimos_60['Close'].pct_change().rolling(window=30).std() * np.sqrt(30),
        "Retorno Acumulado (%)": (1 + df_ultimos_60['Close'].pct_change()).cumprod() - 1,
        "Desviación Estándar 30": df_ultimos_60['Close'].rolling(window=30).std(),
        "RSI": df_ultimos_60['RSI'],
        "MACD": df_ultimos_60['MACD'],
        "ADX": df_ultimos_60['ADX']
    }

    # Mostrar métricas actuales (último valor)
    st.subheader("Valores actuales de KPIs")
    cols = st.columns(len(kpis))
    for i, (label, serie) in enumerate(kpis.items()):
        valor = serie.dropna().iloc[-1] if not serie.dropna().empty else np.nan
        display_val = f"{valor:.4f}" if abs(valor) < 100 else f"{valor:.2f}" if not np.isnan(valor) else "N/A"
        cols[i].metric(label, display_val)

    st.markdown("---")

    # Gráficos de barras para evolución de KPIs seleccionados
    st.subheader("Evolución de KPIs (últimos 60 días)")

    # Selección interactiva de KPIs para mostrar
    opciones_kpis = list(kpis.keys())
    seleccionados = st.multiselect("Selecciona KPIs para graficar", opciones_kpis, default=opciones_kpis[:4])

    for label in seleccionados:
        serie = kpis[label].dropna()
        if serie.empty:
            st.warning(f"No hay datos para {label}")
            continue
        df_plot = pd.DataFrame({'Fecha': serie.index, label: serie.values})
        fig = px.bar(df_plot, x='Fecha', y=label, title=label, labels={'Fecha': 'Fecha', label: label})
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Gráficos de líneas para KPIs técnicos (RSI, MACD, ADX)
    st.subheader("Indicadores Técnicos (últimos 60 días)")
    indicadores_tecnicos = ['RSI', 'MACD', 'ADX']
    for label in indicadores_tecnicos:
        serie = kpis[label].dropna()
        if serie.empty:
            st.warning(f"No hay datos para {label}")
            continue
        df_plot = pd.DataFrame({'Fecha': serie.index, label: serie.values})
        fig = px.line(df_plot, x='Fecha', y=label, title=label, labels={'Fecha': 'Fecha', label: label})
        fig.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
