# proyecto_int_V_2025_1_2
proyecto integrador V primer semestre segundo bloque 

Este proyecto automatiza la recolección, procesamiento y modelado predictivo de datos financieros históricos, integrando un pipeline completo que descarga datos, entrena un modelo de Random Forest para forecasting, y genera predicciones almacenadas para análisis posteriores. Incluye además un flujo de integración continua mediante GitHub Actions para mantener el sistema actualizado y reproducible.

Estructura del proyecto
src/XRP/static/data/ : Datos históricos y archivos CSV procesados

src/XRP/static/models/ : Modelos entrenados serializados (.pkl)

src/XRP/static/predictions/ : Resultados de predicciones guardadas

src/XRP/static/logger.py : Configuración de logging personalizada

src/XRP/collector.py : Clase para descarga y almacenamiento de datos

src/XRP/modeller.py : Clase para entrenamiento, evaluación y predicción del modelo

prueba.py : Script principal para ejecutar entrenamiento y predicción

.github/workflows/update_data.yml : Configuración de GitHub Actions para automatización

Instalación
Se recomienda crear un entorno virtual para aislar dependencias. Para instalar las librerías necesarias, utiliza el archivo requirements.txt que lista las versiones exactas de los paquetes requeridos, asegurando reproducibilidad:

El archivo requirements.txt documenta las dependencias concretas para ejecutar la aplicación. Si en el futuro se distribuyera como paquete, se podría complementar con un setup.py para definir dependencias abstractas y metadatos del paquete.

Uso de GitHub Actions
El workflow definido en .github/workflows/update_data.yml automatiza la actualización del dataset y el entrenamiento del modelo cada vez que se hace push a la rama main. Este proceso incluye:

Checkout del repositorio

Configuración del entorno Python y entorno virtual

Instalación de dependencias

Ejecución del script principal (prueba.py) que realiza la descarga, entrenamiento y predicción

Commit y push automático de los resultados generados (modelos, datos, predicciones)

Esto asegura integración continua y despliegue automatizado sin intervención manual.

Ejecución manual de componentes
Collector: Ejecutar el método update_data() de la clase DataCollector para descargar y almacenar datos históricos actualizados en CSV y SQLite.

Modelo: Usar la clase Modeller para cargar datos, entrenar el modelo con validación temporal, guardar el modelo entrenado y generar predicciones.

Dashboard : Integrar con herramientas de visualización para mostrar resultados y predicciones en tiempo real, utilizando los archivos generados por el collector y el modelo.


GitHub Actions. (2024). Automate, customize, and execute your software development workflows right in your repository with GitHub Actions. GitHub Docs. https://docs.github.com/actions

pandas Development Team. (2023). pandas: Powerful Python data analysis toolkit (Version 1.5) [Computer software]. https://pandas.pydata.org/

scikit-learn developers. (2023). scikit-learn: Machine learning in Python (Version 1.2) [Computer software]. https://scikit-learn.org/

yfinance developers. (2023). yfinance: Yahoo! Finance market data downloader (Version 0.2.18) [Computer software]. https://github.com/ranaroussi/yfinance

Python Software Foundation. (2023). Python programming language (Version 3.13) [Computer software]. https://www.python.org/

SQLite Consortium. (2023). SQLite database engine (Version 3.41) [Computer software]. https://www.sqlite.org/