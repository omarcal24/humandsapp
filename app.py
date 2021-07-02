import streamlit as st
import pandas as pd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Dataset de IBM para predecir desgaste en sus trabajadores",
                   layout='wide')

st.title("Dataset de IBM para predecir desgaste en sus trabajadores")

st.header("Proyecto hecho por Pablo Martín y Omar Calderón")
st.header("Para el Demoday de Saturdays AI edición Bilbao")

df = pd.read_csv('data/IBM-HR.csv')

st.subheader("Primeras columnas del dataset")
st.write(df.head())

st.subheader("Descripción del dataset")
st.write(df.describe())

st.subheader("Valores únicos por columna")



with st.echo():
    columns_list = df.columns.values.tolist()
    for column in columns_list:
        print(column)
        print(df[column].unique())
        print('\n')

#slot1 = st.empty()
#slot2 = st.empty()

col_dict = {}

columns_list = df.columns.values.tolist()
for column in columns_list:
    col_dict[column] = df[column].unique()

#st.write(col_dict)

my_expander = st.beta_expander("Valores únicos por columna", expanded=False)

with my_expander:
    for key, value in col_dict.items():
        st.write(key, value)


#columns_list = df.columns.values.tolist()
#for column in columns_list:
#    slot1.text(column)
#    slot2.text(df[column].unique())


st.subheader("Drop de las columnas que no son necesarias")
#if st.button("Drop a las columnas"):
with st.echo():
    df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], inplace=True, axis=1)


st.subheader("Separamos la variable dependiente de las independientes")
with st.echo():
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

st.subheader("Transformación de X a dummies, X_dummies = pd.get_dummies(X)")
with st.echo():
    X_dummies = pd.get_dummies(X)

st.subheader("Ampliamos el dataset")
with st.echo():
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_dummies, y)


st.write(f'''Shape of X before SMOTE: {X_dummies.shape}
Shape of X after SMOTE: {X_sm.shape}''')
st.write('\nBalance of positive and negative classes (%):')
st.write(y_sm.value_counts(normalize=True) * 100)


st.subheader("Traemos el modelo RandomForestClassifier")
with st.echo():
    clf = RandomForestClassifier()


st.subheader("Hacemos el fit en el modelo y muestra la distribución de probabilidades")
with st.echo():
    clf.fit(X_sm, y_sm)

st.subheader("Distribución de probabilidades")
my_expander3 = st.beta_expander("Mostrar distribución de probabilidades", expanded=False)
with my_expander3:
    fig3 = plt.figure(figsize=(3, 2))
    prob = clf.predict_proba(X_sm)[:,1]
    plt.hist(prob)
    st.pyplot(fig3)


augmented_df = X_sm
augmented_df['Attrition'] = y_sm
augmented_df['Attrition'] = pd.Series(prob)
burnt_df = augmented_df[augmented_df['Attrition'] > 0.6]
candidate_1 = burnt_df.iloc[2]

st.subheader("Mostrar Candidato 1")
st.write("Nivel de Attrition del Candidato 1")
st.write(candidate_1['Attrition'])

st.subheader("Ajustes y cambios en las condiciones del Candidato 1")
candidate_1.drop('Attrition', inplace=True)
with st.echo():
    candidate_1['MonthlyIncome'] = candidate_1['MonthlyIncome'] * 1.1
    candidate_1['OverTime_No'] = 1
    candidate_1['OverTime_Yes'] = 0

candidate_1 = np.array(candidate_1).reshape(1,-1)
prob_candidate_1 = clf.predict_proba(candidate_1)[:,1]
new_attr = prob_candidate_1.astype(np.float64)
if st.button("Mostrar nueva probabilidad de desgaste"):
    st.write("La nueva probabilidad de desgaste del Candidato 1 es:")
    st.write(new_attr[0])
