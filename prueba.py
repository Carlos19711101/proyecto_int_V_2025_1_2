# from src.XRP.static.collector import DataCollector

# if __name__ == "__main__":
#     collector = DataCollector()
#     collector.update_data()
#     print("Datos descargados y actualizados correctamente.")

    
from src.XRP.static.collector import DataCollector
import os
print("Directorio actual:", os.getcwd())

from src.XRP.modeller import Modeller

if __name__ == "__main__":
    # Actualizar datos
    collector = DataCollector()
    collector.update_data()
    print("Datos descargados y actualizados correctamente.")
    
    # Entrenar modelo
    modeller = Modeller()
    modeller.train()
    
    # Predecir
    predictions = modeller.predict()
    print(f"Próximas predicciones: {predictions}")

