# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.ticker as mticker
import warnings
import io
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

if 'nombre_municipio' not in df.columns or df.empty:
    st.error("❌ El archivo no contiene la columna 'nombre_municipio' o está vacío.")
    st.stop()

if 'asegurados' not in df.columns or 'trabajadores_asegurados' not in df.columns:
    st.error("❌ El archivo debe contener las columnas 'asegurados' y 'trabajadores_asegurados'.")
    st.stop()

# Selección de variable
variable_mostrar = st.radio("Selecciona variable a mostrar:", ['Asegurados', 'Trabajadores asegurados'], index=1)

# Selección de municipios
municipios_disponibles = sorted(df['nombre_municipio'].dropna().unique())
col1, col2 = st.columns([4, 1])
with col2:
    seleccionar_todos = st.checkbox("Seleccionar todos", value=True)
with col1:
    if seleccionar_todos:
        municipios_seleccionados = municipios_disponibles
    else:
        municipios_seleccionados = st.multiselect(
            "Selecciona uno o más municipios:",
            options=municipios_disponibles,
            default=[]
        )

if not municipios_seleccionados:
    st.warning("⚠️ Debes seleccionar al menos un municipio para continuar.")
    st.stop()

# Agregado anual
df = df[df['nombre_municipio'].isin(municipios_seleccionados)]
df_anual = df.groupby('año')[[variable_mostrar]].sum().reset_index()
df_anual['crecimiento_%'] = df_anual[variable_mostrar].pct_change() * 100
df_anual['crecimiento_%'] = df_anual['crecimiento_%'].fillna(0).round(2)

X = df_anual[['año']]
y = df_anual[variable_mostrar]
años_futuro = np.arange(df_anual['año'].max() + 1, df_anual['año'].max() + 7)
resultados = pd.DataFrame({'Año': años_futuro})

# Gráfico
fig, ax = plt.subplots(figsize=(12, 6))
color_base = 'black' if variable_mostrar == 'asegurados' else 'purple'
ax.plot(df_anual['año'], y, marker='o', label=f'Histórico ({variable_mostrar})', color=color_base)

for x, y_val in zip(df_anual['año'], y):
    ax.text(x, y_val + max(y)*0.01, f"{int(y_val):,}", ha='center', va='bottom', fontsize=9)

for x, growth in zip(df_anual['año'], df_anual['crecimiento_%']):
    color = 'green' if growth > 0 else ('red' if growth < 0 else 'gray')
    ax.text(x, y[df_anual['año'] == x].values[0] - max(y)*0.05, f"{growth:.1f}%", 
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

# Personalización gráfico
ax.set_title(f"Proyección de {variable_mostrar.replace('_', ' ')}")
ax.set_xlabel("Año")
ax.set_ylabel(variable_mostrar.replace('_', ' ').capitalize())
ax.legend()
ax.grid(True)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
st.pyplot(fig)

# Tabla
municipio_nombre = "Todos los municipios" if seleccionar_todos else ', '.join(municipios_seleccionados)
resultados['Municipio'] = municipio_nombre
st.markdown("### 📋 Tabla de predicciones")

format_dict = {"Año": lambda x: int(x), "Municipio": lambda x: x}
for col in resultados.columns:
    if col not in format_dict:
        format_dict[col] = "{:,.0f}"

styled_df = resultados.style.format(format_dict).set_table_styles([
    {"selector": "th", "props": [("font-weight", "bold"), ("text-align", "center")]}
]).set_properties(**{"text-align": "center"})

st.dataframe(styled_df)

# Botones de descarga
csv = resultados.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Descargar CSV con predicciones",
    data=csv,
    file_name=f'proyecciones_{variable_mostrar}.csv',
    mime='text/csv',
    key="download_csv"
)

output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    resultados.to_excel(writer, index=False, sheet_name='Predicciones')
    df_anual.to_excel(writer, index=False, sheet_name='Histórico')  # Hoja adicional

st.download_button(
    label="⬇️ Descargar Excel con predicciones",
    data=output.getvalue(),
    file_name=f"proyecciones_{variable_mostrar}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="download_excel"
)
