import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from src.XRP.static.logger import setup_logger
from .preprocessor import clean_data  # Asegúrate de tener esta función

class Modeller:
    def __init__(self,
                 data_path='src/XRP/static/data/historical.csv',
                 model_path='src/XRP/static/models/model.pkl'):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.logger = setup_logger('Modeller', 'modeller.log')
        self.feature_cols = ['Open', 'High', 'Low', 'Volume', 'DayOfWeek', 'Month', 'Year']

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        df = clean_data(df)
        df = df.set_index('Date').sort_index()
        return df

    def train(self):
        self.logger.info("Cargando datos para entrenamiento...")
        df = self.load_data()

        X = df[self.feature_cols]
        y = df['Close']

        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        maes = []

        self.logger.info(f"Iniciando validación cruzada temporal con {tscv.get_n_splits()} splits...")

        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            self.logger.info(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            rmses.append(rmse)
            maes.append(mae)

        self.logger.info(f"RMSE promedio: {np.mean(rmses):.4f}, MAE promedio: {np.mean(maes):.4f}")
        print(f"RMSE promedio: {np.mean(rmses):.4f}, MAE promedio: {np.mean(maes):.4f}")

        # Entrenar modelo final con todos los datos
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)

        # Crear carpeta para modelo si no existe
        model_dir = self.model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelo con pickle
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"Modelo guardado correctamente en {self.model_path}")
            self.logger.info(f"Modelo final entrenado y guardado en {self.model_path}")
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            self.logger.error(f"Error al guardar modelo: {e}")

        return final_model

    def predict(self, input_data=None):
        self.logger.info("Cargando modelo para predicción...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"El archivo de modelo no existe: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)

        if input_data is None:
            self.logger.info("No se proporcionaron datos de entrada, usando últimas 30 filas para predecir...")
            df = self.load_data()
            input_data = df.iloc[-30:][self.feature_cols]

        input_data = input_data.ffill().bfill()

        self.logger.info(f"Realizando predicciones para {len(input_data)} muestras...")
        preds = model.predict(input_data)

        return preds

    def predict_and_save(self, input_data=None, output_path='src/XRP/static/predictions/predictions.csv'):
        preds = self.predict(input_data)

        if input_data is not None and hasattr(input_data, 'index'):
            df_preds = pd.DataFrame({'Date': input_data.index, 'Prediction': preds})
        else:
            df_preds = pd.DataFrame({'Prediction': preds})

        output_path = Path(output_path)
        pred_dir = output_path.parent
        pred_dir.mkdir(parents=True, exist_ok=True)

        try:
            df_preds.to_csv(output_path, index=False)
            print(f"Predicciones guardadas correctamente en {output_path}")
            self.logger.info(f"Predicciones guardadas en {output_path}")
        except Exception as e:
            print(f"Error al guardar predicciones: {e}")
            self.logger.error(f"Error al guardar predicciones: {e}")

        return preds
