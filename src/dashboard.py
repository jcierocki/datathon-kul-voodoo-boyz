import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from py2neo import Graph
import visdcc
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import base64
import flask

# nltk.download('popular')

server = flask.Flask(__name__)

app = dash.Dash(url_base_pathname="/", external_stylesheets=[dbc.themes.MINTY], server=server)

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
    df_mov = df_mov.drop_duplicates()[:-1]

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
                        margin=dict(l=50, r=25, t=45, b=25))
    # Set x-axes range
    fig1.update_xaxes(title = 'Number of Artists', 
                    range=[0, max(movement_sum['Count']) + 5])
    fig1.update_yaxes( tickmode='linear',
                        tickfont_size=9)
    fig1.update_layout()
    
    return fig1

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="https://freepngimg.com/download/artwork/141804-photos-picsart-effect-free-transparent-image-hd.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("Artists and their artworks"))
                    ],
                    align="center",
                    className="g-0",
                )
            )
        ]
    )
)

def get_number_artists():
    df_artists = pl.from_dicts(graph.run("""
            match (a: Artist)
            return a.name as artist
            order by artist;
    """).data(), schema={"artist": str}).to_pandas()

    df_artists['artist'] = df_artists['artist'].str.replace('\(after\) ', '')
    # drop row with "and ..."
    df_artists = df_artists['artist'].unique()[:-1]
    return len(df_artists)

def get_number_artworks():
    df_artworks = pl.from_dicts(graph.run("""
            match (a: Artwork)
            return a.name as artwork
            order by artwork;
    """).data(), schema={"artwork": str}).to_pandas()
    # drop row with "and ..."
    df_artworks = df_artworks['artwork'].unique()
    return len(df_artworks)

def get_number_movements():
    df_movements = pl.from_dicts(graph.run("""
            match (a: Movement)
            return a.name as mov
            order by mov;
    """).data(), schema={"mov": str}).to_pandas()
    # drop row with "and ..."
    df_movements = df_movements['mov'].unique()
    return len(df_movements)

row_general_info = html.Div(
    [
        dbc.Row(
            html.Div(html.Br()), style={"text-align": "center", "font-size": "large"}, align="center"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(get_number_artists()), style={"text-align": "center", "font-size": "xx-large", "color": color_2, "font-weight": "bold"}),
                dbc.Col(html.Div(get_number_artworks()), style={"text-align": "center", "font-size": "xx-large", "color": color_2, "font-weight": "bold"}),
                dbc.Col(html.Div(get_number_movements()), style={"text-align": "center", "font-size": "xx-large", "color": color_2, "font-weight": "bold"})
            ], align="center"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("ARTISTS"), style={"text-align": "center", "font-size": "large"}),
                dbc.Col(html.Div("ARTWORKS"), style={"text-align": "center", "font-size": "large"}),
                dbc.Col(html.Div("MOVEMENTS"), style={"text-align": "center", "font-size": "large"}),
            ], align="center"
        )
    ]
)

# NETWORK
def artworks_network():
    df_net = pl.from_dicts(graph.run("""
        match (artwork: Artwork) -- (artist: Artist)
        return artist.name as artist, artwork.name as artwork, artwork.image_url as url
        order by artist;
    """).data(), schema={"artist": str, "artwork": str, "url": str}).to_pandas()
    # Artists
    df_net['artist'] = df_net['artist'].str.replace('\(after\) ', '')
    # Get numbering for artworks
    temp_count = df_net.artist.value_counts(sort = False).reset_index()
    temp_count.columns = ['artist', 'Count']
    temp_count = temp_count.reset_index(drop=True)
    lst_numbers= []
    for i in list(temp_count['Count']):
        for j in range(0, i):
            lst_numbers.append(str(j + 1))
    df_net['Number'] = lst_numbers
    return df_net

df_network = artworks_network()
# Get list of unique artists
unique_artist = df_network['artist'].unique()

## SPECIALIZATION
def specialization():
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
    df_merged = df_spec.merge(spec_desc, on=['spec', 'desc'])
    df_merged.sort_values(by=['artist', 'spec'], inplace=True)
    df_merged = df_merged.reset_index(drop=True).drop_duplicates().reset_index(drop=True)
    return df_merged

merged = specialization()
dropdown_specialization = merged['artist'].unique()



app.layout = dbc.Container(fluid=True, children=[
    navbar,
    row_general_info,
    html.Div([
        dbc.Row(
            html.Div(html.Br()), style={"text-align": "center", "font-size": "large"}, align="center"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div([
                    visdcc.Network(id = 'net', options = dict(height = '550px', width='90%')),
                    dcc.Dropdown(unique_artist, unique_artist[0], id='dropdown_net')
                ])),
                dbc.Col(html.Div([
                            dcc.Graph(id="graph", figure = plot_movements(fig_movements))
                        ]))
            ], align="center"
        )
    ]),
    html.Div([
        dbc.Row(
            html.Div(html.Br()), style={"text-align": "center", "font-size": "large"}, align="center"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div([
                            dbc.Row(
                                html.Div("Artist and their specializations"), style={"text-align": "center", "font-size": "x-large"}, align="center"
                            ),
                            dbc.Row(
                                html.Div([
                                    dcc.Graph(id="spec_plot") ,
                                    dcc.Dropdown(dropdown_specialization, 'Anders Zorn', id='dropdown_spec')
                                ])
                            ),
                ])),
                dbc.Col(html.Div([
                            dbc.Row(
                                html.Div(html.Br()), style={"text-align": "center", "font-size": "large"}, align="center"
                            ),
                            dbc.Row(
                                html.Div("Birthplaces of Artists"), style={"text-align": "center", "font-size": "x-large"}, align="center"
                            ),
                            dbc.Row(
                                html.Iframe(id = "map", srcDoc=open('data/birthplaces_map.html', 'r').read(), width = '50%', height='550'), align="center"
                            ),
                ]))
            ], align="center"
        )
    ]),
    html.Div([
        dbc.Row(
            html.Div(html.Br()), style={"text-align": "center", "font-size": "large"}, align="center"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div([
                            dbc.Row(
                                html.Div("The Most Used Mediums"), style={"text-align": "center", "font-size": "x-large"}, align="center"
                            ),
                            dbc.Row(
                                html.Img(src='data:image/;base64,{}'.format(base64.b64encode(open("data/medium.png", 'rb').read()).decode('ascii'))), align="center"
                            ),
                ])),
                dbc.Col(html.Div([
                            dbc.Row(
                                html.Div("TEXT"), style={"text-align": "center", "font-size": "large"}, align="center"
                            ),
                ]))
            ], align="center"
        )
        

    ])
])

@app.callback(
    Output('net', 'data'),
    [Input("dropdown_net", "value")]
)
def update_network(x):
    nodes = []
    edges = []
    df_painting = df_network.loc[df_network['artist'] == x].reset_index()
    for art, work, url in df_painting[['artist', 'Number', 'url']].itertuples(index=False):
        if not any(d['id'] == art for d in nodes):
            nodes.append({'id': art, 'label': art, 'shape': 'dot', 'size': 5, 'color': color_2})
        if not any(d['id'] == work for d in nodes):
            nodes.append({'id': work, 'label': work, 'shape': 'image', 'image': url,'size': 8, 'color': color_2})
        if not any(d['id'] == art + ' - ' + work for d in edges):
            edges.append({'id': art + ' - ' + work,
            'from': art,
            'to': work,
            'width': 1,
            'color': color_2})
    data = {'nodes': nodes, 'edges': edges}
    return data

@app.callback(
    Output("spec_plot", "figure"),
    [Input("dropdown_spec", "value")]
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

    return fig_spec


if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8000)
