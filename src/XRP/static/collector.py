import os
import pandas as pd
import sqlite3
import yfinance as yf
from .logger import setup_logger

class DataCollector:
    def __init__(self,
                 symbol='XRP-USD',
                 csv_path='src/XRP/static/data/historical.csv',
                 db_path='src/XRP/static/data/historical.db'):
        self.symbol = symbol
        self.csv_path = csv_path
        self.db_path = db_path
        self.logger = setup_logger('DataCollector', 'collector.log')

    def download_data(self):
        self.logger.info(f"Descargando datos históricos para {self.symbol}")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period="max", interval="1d")
        df.reset_index(inplace=True)
        self.logger.info(f"Datos descargados: {len(df)} filas")
        return df

    def save_to_csv(self, df):
        self.logger.info(f"Guardando datos en CSV: {self.csv_path}")
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                old_df = pd.read_csv(self.csv_path, parse_dates=['Date'])
                combined = pd.concat([old_df, df]).drop_duplicates(subset=['Date']).sort_values('Date')
            except pd.errors.EmptyDataError:
                self.logger.warning(f"Archivo CSV vacío o corrupto, se sobrescribe con nuevos datos.")
                combined = df
        else:
            combined = df
        combined.to_csv(self.csv_path, index=False)
        self.logger.info(f"CSV actualizado con {len(combined)} filas")

    def save_to_sqlite(self, df):
        self.logger.info(f"Guardando datos en SQLite: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical (
                Date TEXT PRIMARY KEY,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                Dividends REAL,
                StockSplits REAL
            )
        ''')
        conn.commit()

        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO historical (Date, Open, High, Low, Close, Volume, Dividends, StockSplits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['Date'].strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'],
                  row['Close'], row['Volume'], row['Dividends'], row['Stock Splits']))
        conn.commit()
        conn.close()
        self.logger.info(f"SQLite actualizado con {len(df)} filas")

    def update_data(self):
        df = self.download_data()
        self.save_to_csv(df)
        self.save_to_sqlite(df)
        self.logger.info("Actualización de datos completada")
