import dash
import visdcc
import pandas as pd
import polars as pl
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from py2neo import Graph
from dash import dash_table

app = dash.Dash(__name__)

nodes = []
edges = []
color_2 = '#0F7620'

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# ARTISTS AND ARTWORKS
df_network = pl.from_dicts(graph.run("""
    match (artwork: Artwork) -- (artist: Artist)
    return artist.name as artist, artwork.name as artwork, artwork.image_url as url
    order by artist;
""").data(), schema={"artist": str, "artwork": str, "url": str}).to_pandas()
# Artists
df_network['artist'] = df_network['artist'].str.replace('\(after\) ', '')
# Get numbering for artworks
temp_count = df_network.artist.value_counts(sort = False).reset_index()
temp_count.columns = ['artist', 'Count']
temp_count = temp_count.reset_index(drop=True)
lst_numbers= []
for i in list(temp_count['Count']):
    for j in range(0, i):
        lst_numbers.append(str(j + 1))
df_network['Number'] = lst_numbers
# Get list of unique artists
unique_artist = df_network['artist'].unique()

app.layout = html.Div([
    visdcc.Network(id = 'net', options = dict(height = '600px', width='50%')),
    dcc.Dropdown(unique_artist, unique_artist[0], id='dropdown')
])


@app.callback(
    Output('net', 'data'),
    [Input("dropdown", "value")]
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

app.run_server(debug=True)
