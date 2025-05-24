import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el DataFrame para entrenamiento y predicción.
    Elimina columnas problemáticas y rellena valores faltantes.
    """
    cols_to_drop = ['Dividends', 'StockSplits', 'SPY_Close']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Rellenar valores faltantes sin warnings
    df = df.ffill().bfill()

    df = df.dropna()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    return df
