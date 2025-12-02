# Estimador-de-esperanza-de-vida
Proyecto de ciencia de los datos 2025-2
# Autores
1. Sara Mariana Sanabria Cruz - C.C. 1085898875
2. Santiago Alejandro Cardona Escobar - C.C. 1053862888
3. Santiago Cuesta Ocampo - C.C. 1002652863
   
# Descripción del proyecto
Este proyecto desarrolla una aplicación web para estimar la esperanza de vida usando datos históricos de la OMS, a partir del dataset de Kaggle “Life Expectancy (WHO)”.

# Como ejecutar el proyecto en Visual Studio Code

## 1. Requisitos previos
1. Python 3.10 o superior instalado.
Comprobar en PowerShell o CMD:
python --version

## 2. Abrir el archivo de la API.
1. En la carpeta Backend, abrir el archivo: lifeapi.py

## 3. Instalar dependencias
En la terminal (con el entorno activado):
pip install flask flask-cors pandas numpy scikit-learn joblib

## 4. Ejecutar código en VSC
Run python file

## 5. Abrir el frontend en el navegador
Abrir el HTML directamente 
En la carpeta frontend, abrir el archivo: panel_esperanza_vida.html

## 6. Uso de la aplicación
En la página que se abrió en el paso 6:
1. Seleccionar un país.
2. Definir el año (por defecto 2025, dentro del rango 2000–2030).
3. Completar las demás variables usando los rangos sugeridos como guía.
4. Hacer clic en “Calcular esperanza de vida”.
5. Se mostrarán:
5.1 Un valor numérico en “Regresión lineal”.
5.2 Una categoría cualitativa en “KNN”.
5.3 Una categoría cualitativa en “MLP”.
