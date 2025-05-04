from src.XRP.static.collector import DataCollector

if __name__ == "__main__":
    collector = DataCollector()
    collector.update_data()
    print("Datos descargados y actualizados correctamente.")
