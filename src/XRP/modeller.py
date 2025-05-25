import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from src.XRP.static.logger import setup_logger
from .preprocessor import clean_data

class Modeller:
    def __init__(self,
                 data_path='src/XRP/static/data/historical.csv',
                 model_path='src/XRP/static/models/model.pkl'):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.logger = setup_logger('Modeller', 'modeller.log')
        self.feature_cols = ['Open', 'High', 'Low', 'Volume', 'DayOfWeek', 'Month', 'Year']

    def load_data(self):
        try:
            print(f"Cargando datos desde: {self.data_path.resolve()}")
            df = pd.read_csv(self.data_path, parse_dates=['Date'])
            df = clean_data(df)
            df = df.set_index('Date').sort_index()
            print(f"Datos cargados correctamente, filas: {len(df)}")
            
            processed_path = self.data_path.parent / 'procesados.csv'
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=True)
            print(f"Archivo procesado guardado en: {processed_path.resolve()}")
            
            return df
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            raise

    def train(self):
        self.logger.info("Cargando datos para entrenamiento...")
        df = self.load_data()

        X = df[self.feature_cols]
        y = df['Close']

        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        maes = []
        resultados = []

        self.logger.info(f"Iniciando validación cruzada temporal con {tscv.get_n_splits()} splits...")

        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            resultados.append({'Fold': fold, 'RMSE': rmse, 'MAE': mae})

            self.logger.info(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            rmses.append(rmse)
            maes.append(mae)

        rmse_promedio = np.mean(rmses)
        mae_promedio = np.mean(maes)
        resultados.append({'Fold': 'Promedio', 'RMSE': rmse_promedio, 'MAE': mae_promedio})

        df_resultados = pd.DataFrame(resultados)
        entrenado_path = self.data_path.parent / 'entrenado.csv'
        entrenado_path.parent.mkdir(parents=True, exist_ok=True)
        df_resultados.to_csv(entrenado_path, index=False)
        print(f"Resultados de entrenamiento guardados en: {entrenado_path.resolve()}")

        self.logger.info(f"RMSE promedio: {rmse_promedio:.4f}, MAE promedio: {mae_promedio:.4f}")
        print(f"RMSE promedio: {rmse_promedio:.4f}, MAE promedio: {mae_promedio:.4f}")

        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        print(f"Modelo entrenado con {len(X)} muestras")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"Modelo guardado correctamente en {self.model_path.resolve()}")
            self.logger.info(f"Modelo guardado en {self.model_path}")
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            raise

        return final_model

    def predict(self, input_data=None):
        self.logger.info("Cargando modelo para predicción...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {self.model_path.resolve()}")

        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            raise

        if input_data is None:
            self.logger.info("Usando últimas 30 filas para predecir...")
            df = self.load_data()
            input_data = df.iloc[-30:][self.feature_cols]

        input_data = input_data.ffill().bfill()
        print(f"Datos procesados para predicción:\n{input_data}")
        
        preds = model.predict(input_data)
        print(f"Predicciones generadas: {preds}")
        return preds

    def predict_and_save(self, input_data=None):
        preds = self.predict(input_data)

        base_path = self.data_path.parent
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar predicciones generales (todas)
        pred_gen_path = base_path / 'predicciones_generales.csv'
        df_pred_gen = pd.DataFrame({'Prediction': preds})
        df_pred_gen.to_csv(pred_gen_path, index=False)
        print(f"Archivo predicciones_generales.csv guardado en {pred_gen_path.resolve()}")

        # Guardar próximas predicciones (últimas 30)
        prox_pred_path = base_path / 'proximas_predicciones.csv'
        ultimas_30 = preds[-30:] if len(preds) >= 30 else preds
        df_prox_pred = pd.DataFrame({'Next_Prediction': ultimas_30})
        df_prox_pred.to_csv(prox_pred_path, index=False)
        print(f"Archivo proximas_predicciones.csv guardado en {prox_pred_path.resolve()}")

        return preds

    def load_existing_predictions(self, path='src/XRP/static/predictions/prediction.csv'):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path.resolve()}")
        try:
            df = pd.read_csv(path, parse_dates=['Date'])
            print(f"Predicciones cargadas desde {path.resolve()}")
            return df
        except Exception as e:
            print(f"Error al cargar predicciones: {e}")
            raise
