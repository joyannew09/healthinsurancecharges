import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #turn categorical data in numerical data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

# Intialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Load Data and Preprocessing of Data
#filename = 'insurance-Copy1.csv'
df = pd.read_csv('insurance-Copy1.csv') #load file to dataframe

# Create Single/family status
df['single_family'] = np.where(df.children == 0, 'single', 'family')

# Create age groups
binsAge = [18, 35, 50, 65]
labelsAge = ['[18-34]', '[35-49]', '50+']
df['agegroup'] = pd.cut(df['age'], bins=binsAge, labels=labelsAge, right=False)

# Create BMI groups

binsBMI = [0, 19, 25, 30, 55]
labelsBMI = ['Underweight', 'Normal Weight', 'Overweight', 'Obesity']
df['bmigroup'] = pd.cut(df['bmi'], bins = binsBMI, labels = labelsBMI, right = False)

# Create Machine Learning Model

# Encode categorical data
encoder = LabelEncoder()
cat_cols = [col for col in df.columns if df[col].dtype in ('O', 'category')]
df_model = df.copy()
for col in cat_cols:
    df_model[col] = encoder.fit_transform(df_model[col])

# Train the Model
listTrainCols = ['age', 'bmi', 'sex', 'children', 'smoker', 'region']

X = df_model[listTrainCols]
y = df_model['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)
model = LinearRegression()
model.fit(X_train, y_train)

# APP Layout
app.layout = html.Div([
    html.H1('Health Insurance Charge Analysis Dashboard', style = {'testAlign': 'center'}),

    dcc.Dropdown(
        id = 'plot-selector',
        options = [
            {'label': 'Charge by Family Status and Smoking','value': 'family_smoke'},
            {'label': 'Charge by Number of Children','value': 'children'},
            {'label': 'Average Charge by Age Group','value': 'age_group'},
        ],
        value = 'family_smoke'

    ),

    #Building the Graph
     dcc.Graph(id = 'main graph'),

    # Prediction Section
    html.Div([
        html.H2('Insurance Charge Prediction'),

        #Input feilds
        html.Div([
            html.Label('Age'),
            dcc.Input(id = 'age-input', type = 'number', value = 18), # value = default value

            html.Label('BMI'),
            dcc.Input(id = 'bmi-input', type = 'number', value = 25),

            html.Label('Children'),
            dcc.Input(id = 'children-input', type = 'number', value = 0),

            html.Label('Sex'),
            dcc.Dropdown(
                id = 'sex-input',
                options= [{'label':'Male', 'value':1},
                          {'label':'Female', 'value':0}],
                value = 1,
            ),

            html.Label('Smoker'),
            dcc.Dropdown(
                id = 'smoker-input',
                options = [{'label':'Yes', 'value':1},
                           {'label':'No', 'value':0}],

                value = 0,

            ),

            html.Label('Region'),
            dcc.Dropdown(
                id = 'region-input',
                options = [{'label':'Northeast', 'value':0},
                           {'label':'Northwest', 'value':1},
                           {'label':'Southeast', 'value':2},
                           {'label':'Southwest', 'value':3}],
                value = 0,

            ),
            
        ], style = {'display':'flex', 'flexDirection':'column', 'gap':'15px'} ),

        html.Button('Predict', id = 'predict-button', n_clicks = 0),

        html.Div(id = 'prediction-output')
    

    ]),

])

#Callback function

@app.callback(
    Output('main graph', 'figure'),
    Input('plot-selector', 'value')
)
def update_graph(selected_plot):
    if selected_plot == 'family_smoke':
        fig = px.box(df, y = 'charges', x = 'single_family', color = 'smoker', 
                     title = 'Health Insurance Charges by Family Status and Smoking Habits')
    elif selected_plot == 'children':
        fig = px.box(df, y= 'charges', x = 'children', color = 'smoker',
                     title = "Health Insurance Charges by Number of Children")
    elif selected_plot == 'age_group':
        age_group_charges = df.groupby('agegroup')['charges'].mean().reset_index()
        fig = px.bar(age_group_charges, y= 'charges', x = 'agegroup',
                     title = "Average Health Insurance Charges by Age Group")

    return fig
        
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('age-input', 'value'),
     State('bmi-input', 'value'),
     State('children-input', 'value'),
     State('sex-input', 'value'),
     State('smoker-input', 'value'),
     State('region-input', 'value')
    ])

def predict_charge(n_clicks, age, bmi, children, sex, smoker, region):
    if n_clicks > 0:
        input_data = np.array([[age, bmi, children, sex, smoker, region]])
        prediction = model.predict(input_data)[0]
        return f'Predicted Insurance Charge: ${prediction:,.2f}'
    return ''


if __name__ == '__main__':
    app.run(debug = True)
                







