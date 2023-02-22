import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from py2neo import Graph
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('popular')


app = dash.Dash(__name__)

color_1 = '#CCEECD'
color_2 = '#0F7620'

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

## SPECIALIZATION
df_spec = pl.from_dicts(graph.run("""
    match (s: Specialization) -- (a: Artist)
    return a.name as artist, s.name as spec, s.description as desc
    order by artist
""").data(), schema={"artist": str, "spec": str, "desc": str}).to_pandas()
df_spec['artist'] = df_spec['artist'].str.replace('\(after\) ', '')
df_spec.sort_values(by=['artist', 'spec'], inplace=True)
df_spec = df_spec.reset_index(drop=True).drop_duplicates().reset_index(drop=True)
df_spec = df_spec[:-4]

spec_desc = df_spec[['spec', 'desc']]
spec_desc.sort_values(by=['spec', 'desc'], inplace=True)
spec_desc = spec_desc.reset_index(drop=True).drop_duplicates().reset_index(drop=True)
result = [x for x in spec_desc['desc']]
i = 0
for res in result:
    sentence = sent_tokenize(res)[0]
    result[i] = sentence.replace("investigation\')[1]", "investigation\')").replace("process,[1]", "process,").replace("write\')[1]", "write')").replace("-⁠TAH-;[1]", "-⁠TAH-;").replace("[maŋga])[a]", "[maŋga])").replace("\"matrix\"[citation needed]", "\"matrix\"").replace("verse,[note 1]", "verse,").replace("rhythmic[1][2][3]", "rhythmic")   
    i = i + 1

spec_desc['short'] = result
merged = df_spec.merge(spec_desc, on=['spec', 'desc'])
merged.sort_values(by=['artist', 'spec'], inplace=True)
merged = merged.reset_index(drop=True).drop_duplicates().reset_index(drop=True)

dropdwn = merged['artist'].unique()


app.layout = html.Div([
    dcc.Dropdown(dropdwn, 'Anders Zorn', id='dropdown'),
    dcc.Graph(id="graph")    
])


@app.callback(
    Output("graph", "figure"),
    [Input("dropdown", "value")]
)
def update_artist(x):
    selected = merged.loc[merged['artist'] == x].reset_index()
    selected['value'] = 100/len(selected)
    color_selected = [color_1]
    if len(selected) > 1:
        color_selected = get_color_gradient(color_1, color_2, len(selected))
    
    fig_spec = go.Figure(data=[go.Pie(labels=selected['spec'],
                            values=selected['value'],
                            hole=.4, 
                            text = selected['short'].str.wrap(30).apply(lambda x: x.replace('\n', '<br>')),
                            textinfo = 'label',
                            hoverinfo = 'text',
                            marker = dict(colors=color_selected))])
    fig_spec.update_layout(title=selected['artist'][0], title_font_size = 20, title_x=0.5, showlegend=False)
    fig_spec.show()
    return fig_spec


app.run_server(debug=True)
