import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import cv2
import numpy as np
# import base64
# import io
# import datetime

# import pandas as pd

# dash.register_page(__name__, path="/ml")
app = dash.Dash(__name__, url_base_pathname="/ml/", external_stylesheets = [dbc.themes.BOOTSTRAP, "https://github.com/plotly/dash-app-stylesheets/blob/master/dash-analytics-report.css"])

app.layout = dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                html.H1("ML model GUI", style={"text-align": "center"})
            ], width={"size": 4, "offset": 4}),
            dbc.Col([
                html.A("Back to main page", href="http://127.0.0.1:8000/")
            ], width={"size": 1}, style={"text-align": "right"})
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Input"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={
                        'width': '90%',
                        'height': '100px',
                        'lineHeight': '100px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '0px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
            ], width={"size": 2, "offset": 3}),
            dbc.Col([
                dbc.Row([
                    html.Div("Some text")
                ]),
                dbc.Row([
                    html.Div("Some text")
                ])
            ], width={"size": 4, "offset": 0})
        ])
    ])

def load_and_preprocess(image):
   image = cv2.imread(image, cv2.IMREAD_GRAYSCALE).resize((512, 512))

   return image

# def np_array_normalise(test_image):
#    np_image = np.array(test_image)
#    np_image = np_image / no_of_pixels
#    final_image = np.expand_dims(np_image, 0)
#    return final_image

# @app.callback(Output('output-prediction', 'children'),
#               Input('upload-image', 'filename'))

# def prediction(image):
#     if image is None:
#         raise dash.exceptions.PreventUpdate()
#     final_img = load_and_preprocess(image)
#     final_img = np_array_normalise(final_img)
#     Y = model.predict(final_img)
#     return Y


if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
