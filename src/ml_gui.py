import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import joblib
import cv2
import base64
import numpy as np

from sklearn.svm import SVC
from PIL import Image

from img_db import preproc_img_svm
from Yolov8_single import YoloV8Classifier, YOLO
from GFNet_single import GFNetClassifier, torch

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
                    multiple=True
                ),
            ], width={"size": 2, "offset": 3}),
            dbc.Col([
                dbc.Row(html.H3("Model results")),
                dbc.Row(class_name="model-output", children=[
                    dbc.Col([html.Div("ML fake detector")]),
                    dbc.Col([html.Div(id="mod-out-1", children=["upload an image"])])
                ]),
                dbc.Row(class_name="model-output", children=[
                    dbc.Col([html.Div("DL fake detector")]),
                    dbc.Col([html.Div(id="mod-out-2", children=["upload an image"])])
                ]),
                dbc.Row(class_name="model-output", children=[
                    dbc.Col([html.Div("DL movement classifier")]),
                    dbc.Col([html.Div(id="mod-out-3", children=["upload an image"])])
                ]),
            ], width={"size": 4, "offset": 0})
        ], style={"padding-top": "10px"})
    ])

mod_fake_svm: SVC = joblib.load("output/svm_trained.xz")
mod_movements_yolo: YOLO = YOLO("weights/best.pt")
mod_gfnet = torch.load("weights/GFNet.pth")
mod_gfnet.eval()

# mod_fake_gfnet = joblib.load("../output/svm_trained.xz")

# def load_image(image):
#    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE).resize((512, 512))

#    return image

# def process_image_svm(image):

# @app.callback(Output('output-prediction', 'children'),
#               Input('upload-image', 'filename'))

# def prediction(image):
#     if image is None:
#         raise dash.exceptions.PreventUpdate()
#     final_img = load_and_preprocess(image)
#     final_img = np_array_normalise(final_img)
#     Y = model.predict(final_img)
#     return Y
@app.callback(
        [
            Output("mod-out-1", "children"),
            Output("mod-out-2", "children"),
            Output("mod-out-3", "children"),
        ],
        Input("upload-data", "contents")
)
def predict_all(image):
    if image is None:
         raise dash.exceptions.PreventUpdate()

    encoded_data = image[0].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    X = preproc_img_svm(img)
    
    print("predicting SVM")
    pred_svm = mod_fake_svm.predict(X)[0]

    img_pil = Image.fromarray(img)

    print("predicting Yolo")
    pred_yolo = YoloV8Classifier().predict(mod_movements_yolo, img_pil)['predictions']
    scores_yolo = [p['probability'] for p in pred_yolo]
    print(scores_yolo)
    point_pred_yolo = [p['movement'] for p in pred_yolo if p['probability'] == max(scores_yolo)][0]
    print(point_pred_yolo)

    print("predicting GFNet")
    pred_gfnet = GFNetClassifier().predict(mod_gfnet, img_pil)['predictions'][0]['type']
    print("done")

    return "Real" if pred_svm == 1 else "Fake", pred_gfnet, point_pred_yolo



if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
