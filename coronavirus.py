# Librerias
import requests # Para hacer la petición a la api
import streamlit as st # Libreria para hacer el dashboard
import numpy as np
import pandas as pd
# Librerias para graficación
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
# Libreria para calcular el error de la predicción
from sklearn.metrics import r2_score, mean_squared_error


LESS_THAN = 1
GREATER_EQUAL = 0


@st.cache
def getAndCleanData():
    # Se definen las columnas de importancia
    cols = ['Country_Region', 'Lat', 'Long', 'Province_State']
    classifier = ['Confirmed', 'Deaths', 'Recovered']
    # Obtenemos los datos
    r = requests.get("http://64.227.56.58:5000/covid19").json()
    tabla = pd.DataFrame(r)
    agrupacion = tabla.groupby(cols)
    frames = []
    for name, group in agrupacion:
        df = group.pivot(index='Classifier', columns='Date', values='Value')
        newCols = []
        df.fillna(0, inplace=True)
        for c in classifier:
            # si no se encuentra el clasificador
            if not c in list(df.index.values):
                # llena todo el renglon con ceros
                df.loc[c] = 0
        # Calculamos los casos existentes actualmente
        existing = pd.Series([int(df.iloc[0, i]) - (int(df.iloc[2, i]) + int(df.iloc[1, i]))
                              for i in range(df.shape[1])]).rename('Existing')
        existing.index = list(df.columns.values)
        df = df.append(existing)
        # Se agregan las columnas Pais_region, Lat, Long, Provincia_estado
        for i in range(len(name)):
            df.insert(i, cols[i], name[i])
        frames.append(df)
    resultado = pd.concat(frames)
    resultado.fillna(0, inplace=True)
    resultado = resultado.sort_index(axis=1)
    return resultado, r


def similarChart(compareDf, dataframe, condition=GREATER_EQUAL):
    # sumamos los casos confirmados por pais
    confirmed = dataframe.loc['Confirmed', :].groupby('Country_Region')
    best = None
    bestDf = None
    # para cada fila comparamos desde que hubo un primer caso
    for name, group in confirmed:
        row = group.sum().drop(
            ['Lat', 'Long', 'Province_State', 'Country_Region'], axis=0)
        temp = row.loc[lambda x: x != 0]
        # se saca la transpuesta de cada uno y se reinician los indices
        trans1 = compareDf.transpose().reset_index()
        trans2 = temp.transpose().reset_index()
        # Determinamos la condición
        if condition == GREATER_EQUAL:
            cond = temp.shape[0] >= compareDf.shape[0]
            size = trans1.shape[0]
        else:
            cond = temp.shape[0] < compareDf.shape[0]
            size = trans2.shape[0]
        # Si el numero de dias con casos es mayor al comparado,
        if cond:
            # se suma la diferencia para cada columna
            for i in range(size):
                dif = 0
                dif += int(trans1.iloc[i, 1]) - int(trans2.iloc[i, 1])
            # si la diferencia en total es la menor,.convert_objects(convert_numeric=True) se guarda como preferida
            if (best == None and dif != 0) or (best != None and abs(dif) < best and dif != 0):
                best = dif
                bestDf = temp.append(
                    pd.Series([name], index=['Country_Region']))
    return bestDf

# Tab Analisis Global


def tabGeneralAnalisis(resultado, r):
    # Obtenemos los casos confirmados de la fecha mas reciente
    forMap = resultado.loc['Confirmed', :]
    lastDate = str(list(forMap.columns.values)[-5])
    forMap = resultado.loc[lambda x: x[lastDate] > 0, :]
    st.write("### Mapa en función de los casos confirmados")
    st.plotly_chart(px.scatter_geo(forMap, lat="Lat", lon="Long", text="Country_Region",
                                   hover_name="Country_Region", hover_data=["Country_Region", "Province_State", lastDate], size=lastDate,
                                   projection="natural earth", size_max=30))
    # Distribucción
    propChart = resultado.drop(['Country_Region','Lat','Long','Province_State'], axis=1).reset_index().groupby('Classifier').sum().drop('Confirmed').reset_index()
    propChart = pd.melt(propChart, id_vars='Classifier', value_vars=list(propChart.columns[1:]), var_name='Date', value_name='Value')

    st.write('### Distribucción de casos confirmados en el mundo')
    st.altair_chart(alt.Chart(propChart).mark_area().encode(
        x="Date:T",
        y=alt.Y("Value:Q", stack="normalize"),
        color="Classifier:N"
    ))


def tabAnalisisByCountry(resultado):
    st.write('### Incremento en casos por pais')
    country = st.selectbox('Selecciona', resultado.Country_Region.unique())
    # Calculos para la grafica
    forChart = resultado.groupby('Country_Region').get_group(country).drop(
        ['Country_Region', 'Lat', 'Long', 'Province_State'], axis=1).groupby(level=0).sum().loc[:, lambda x: x.iloc[0] != 0]
    compare = forChart.loc[['Confirmed']]
    forChart = forChart.loc[['Deaths', 'Recovered', 'Existing']]
    forChart.index = ['Deaths', 'Recovered', 'Existing']
    st.bar_chart(forChart.transpose())
    st.write('### Parecido a :')
    # Calculos para la similitud
    similar = similarChart(forChart.sum(), resultado)
    similarCountry = str(similar['Country_Region'])
    st.write('#### ' + similarCountry)
    similar = similar.drop(
        'Country_Region').reset_index().drop('index', axis=1)
    compare = compare.transpose().reset_index().drop('index', axis=1)
    similar = similar.join(compare)
    similar.columns = [similarCountry, country]

    st.line_chart(similar)
    st.write('### Predicciones polinomiales')
    forecast = forecastByCountry(similar[country], 2, 10, 0).join(similar[country])
    # st.line_chart(forecast)
    fig = go.Figure()
    for colName, col in forecast.iteritems():
        fig.add_trace(go.Scatter(x=list(col.index.values),
                                 y=col, mode='lines+markers', name=colName))
    st.plotly_chart(fig)


def forecastByCountry(country, minElev, maxElev, inter = 20):
    bestModel = None
    bestScore = None
    bestMSE = None
    country.dropna(inplace=True)
    for i in range(maxElev - minElev):
        # Dataframe before reach 20 cases
        newDf = country.loc[lambda x: x >= inter]
        start = country.loc[lambda x: x < inter].shape[0]
        x = range(start, start + newDf.shape[0])
        y = newDf.to_numpy()
        f = np.polyfit(x, y, i + minElev)
        p = np.poly1d(f)
        new_y = p(np.linspace(start, newDf.shape[0] + 5 + start, newDf.shape[0] + 5 + start))
        score = r2_score(y, p(x))
        mse = mean_squared_error(y, p(x))
        if bestScore == None or bestScore > score:
            bestScore = score
            bestModel = np.concatenate((p(range(start)),  new_y), axis=None)
    return pd.DataFrame(bestModel, columns=['Best Forecast'])


# Script principal
resultado, r = getAndCleanData()
opcion = st.sidebar.selectbox(
    'Escala de Visualización', ['Global', 'Por País'])

if opcion == 'Global':
    tabGeneralAnalisis(resultado, r)
elif opcion == 'Por País':
    tabAnalisisByCountry(resultado)
