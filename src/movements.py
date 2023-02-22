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

## MOVEMENTS
fig_movements = go.Figure()
def plot_movements(fig1):
    df_mov = pl.from_dicts(graph.run("""
        match (a: Artist) -- (m: Movement)
        return a.name as artist, m.name as movement
        order by artist;
    """).data(), schema={"artist": str, "movement": str}).to_pandas()

    df_mov['artist'] = df_mov['artist'].str.replace('\(after\) ', '')
    # drop row with "and ..."
    df_mov = df_mov[:-1]

    movement_sum = df_mov.movement.value_counts().reset_index()
    movement_sum.columns = ['Movement', 'Count']
    movement_sum.sort_values(by=['Count', 'Movement'], inplace=True, ascending = [True, False])
    movement_sum = movement_sum.reset_index(drop=True)

    color = movement_sum['Count'].value_counts().reset_index()
    color.columns = ['Color', 'Count']
    my_color_temp= get_color_gradient(color_1, color_2, len(color))

    a = 0
    my_color = []
    for i in list(color['Count']):
        my_color += i * [my_color_temp[a]]
        a = a + 1

    # Draw points
    fig1.add_trace(go.Scatter(x = movement_sum["Count"], 
                            y = movement_sum["Movement"],
                            mode = 'markers',
                            marker = dict(color=my_color),
                            marker_size  = 6))
                            #text = movement_sum['new'], hoverinfo = 'y + x + text'))
    # Draw lines
    for i in range(0, len(movement_sum)):
        fig1.add_shape(type='line',
                        x0 = 0, y0 = i,
                        x1 = movement_sum["Count"][i],
                        y1 = i,
                        line=dict(color=my_color[i]))
    # Set title
    fig1.update_layout(title_text = "Artistic Movements",
                        title_font_size = 20,
                        title_x=0.5,
                        plot_bgcolor = 'rgb(242,242,242)',
                        autosize=False,
                        width=800,
                        height=650,
                        margin=dict(l=25, r=25, t=45, b=25))
    # Set x-axes range
    fig1.update_xaxes(title = 'Number of Artists', 
                    range=[0, max(movement_sum['Count']) + 5])
    fig1.update_yaxes(title = 'Movemenet',
                        tickmode='linear',
                        tickfont_size=9)
    fig1.update_layout()
    fig1.show()
    return fig1



app.layout = html.Div([
    dcc.Graph(id="graph", figure = plot_movements(fig_movements))    
])




app.run_server(debug=True)
