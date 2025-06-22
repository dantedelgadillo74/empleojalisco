
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Proyección Asegurados", layout="centered")
st.title("📈 Proyección de Asegurados - Jalisco")

modelos_seleccionados = st.multiselect(
    "Selecciona modelos de pronóstico a mostrar:",
    ["Lineal", "Ridge", "ARIMA"],
    default=["Lineal", "Ridge", "ARIMA"]
)

df = pd.read_csv("jalisco_asegurados.csv")
df['fecha'] = pd.to_datetime(df['fecha'])
df['año'] = df['fecha'].dt.year

municipios_disponibles = df['nombre_municipio'].unique()
municipios_seleccionados = st.multiselect(
    "Selecciona uno o más municipios:",
    options=municipios_disponibles,
    default=[municipios_disponibles[0]]
)

df = df[df['nombre_municipio'].isin(municipios_seleccionados)]
df_anual = df.groupby('año')['asegurados'].sum().reset_index()
df_anual['crecimiento_%'] = df_anual['asegurados'].pct_change() * 100
df_anual['crecimiento_%'] = df_anual['crecimiento_%'].fillna(0).round(2)

X = df_anual[['año']]
y = df_anual['asegurados']
años_futuro = np.arange(df_anual['año'].max() + 1, df_anual['año'].max() + 7)
resultados = pd.DataFrame({'Año': años_futuro})

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_anual['año'], y, marker='o', label='Histórico', color='black')

for i, (x, y_val) in enumerate(zip(df_anual['año'], df_anual['asegurados'])):
    ax.text(x, y_val + max(y)*0.01, f"{int(y_val):,}", ha='center', va='bottom', fontsize=9)

for i, (x, growth) in enumerate(zip(df_anual['año'], df_anual['crecimiento_%'])):
    color = 'green' if growth > 0 else ('red' if growth < 0 else 'gray')
    ax.text(x, df_anual['asegurados'].iloc[i] - max(y)*0.05, f"{growth:.1f}%", 
            ha='center', va='top', fontsize=9, color=color)

if "Lineal" in modelos_seleccionados:
    model = LinearRegression().fit(X, y)
    pred = model.predict(años_futuro.reshape(-1, 1))
    ax.plot(años_futuro, pred, '--o', label='Lineal', color='blue')
    resultados["Lineal"] = pred
    for x, val in zip(años_futuro, pred):
        ax.text(x, val + max(y)*0.01, f"{int(val):,}", ha='center', va='bottom', fontsize=9, color='blue')

if "Ridge" in modelos_seleccionados:
    model = Ridge().fit(X, y)
    pred = model.predict(años_futuro.reshape(-1, 1))
    ax.plot(años_futuro, pred, '--o', label='Ridge', color='orange')
    resultados["Ridge"] = pred
    for x, val in zip(años_futuro, pred):
        ax.text(x, val + max(y)*0.01, f"{int(val):,}", ha='center', va='bottom', fontsize=9, color='orange')

if "ARIMA" in modelos_seleccionados:
    serie = pd.Series(y.values, index=df_anual['año'])
    model = ARIMA(serie, order=(1, 1, 1)).fit()
    pred = model.forecast(steps=6)
    ax.plot(años_futuro, pred, '--o', label='ARIMA', color='green')
    resultados["ARIMA"] = pred.values
    for x, val in zip(años_futuro, pred):
        ax.text(x, val + max(y)*0.01, f"{int(val):,}", ha='center', va='bottom', fontsize=9, color='green')

ax.set_title("Proyección de asegurados")
ax.set_xlabel("Año")
ax.set_ylabel("Asegurados")
ax.legend()
ax.grid(True)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

st.pyplot(fig)
st.markdown("### 📋 Tabla de predicciones")
st.dataframe(resultados.style.format(precision=0, thousands=","))

csv = resultados.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Descargar CSV con predicciones",
    data=csv,
    file_name='proyecciones_asegurados.csv',
    mime='text/csv'
)

import io
from tempfile import NamedTemporaryFile
import xlsxwriter

output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    resultados.to_excel(writer, sheet_name='Predicciones', index=False)
st.download_button(
    label="⬇️ Descargar Excel con predicciones",
    data=output.getvalue(),
    file_name="proyecciones_asegurados.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
