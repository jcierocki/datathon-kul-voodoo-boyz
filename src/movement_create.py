import pandas
import pyarrow
import os
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote, urlsplit, urlunsplit, unquote
import polars as pl

import pandas as pd
from py2neo import Graph
YOUR_PASSWORD = "password"
YOUR_PORT = 7687

movement_names = ['Ashcan School', 'Pre-Raphaelite Brotherhood', 'Synthetism', 'Tonalism', 'Symbolism', 'Intimism', 'Newlyn School', 'Regionalism', 'Modernism', 'Les Nabis', 'Academism', 'Divisionism', 'Renaissance', 'Abstract', 'Art Nouveau', 'Romanticism', 'Hudson River School', 'Decadent Movement', 'Munich School', 'Mannerism', 'Rocky Mountain School', 'Bristol School', 'Gothic', 'Neoclassicism', 'Avant-Garde', 'Baroque', 'Venetian School',
                  'Realism', 'Delft School', 'Norwich School', 'Luminism', 'Expressionism', 'Dutch Golden Age', 'Heidelberg School', 'Pointillism', 'Utagawa School', 'Purismo', 'Vienna Secession', 'Suprematism', 'Primitivism', 'Surrealism', 'Aestheticism', 'Early Netherlandish', 'Post-Impressionism', 'Impressionism', 'Futurism', 'Dusseldorf School', 'Berlin Secession', 'Barbizon School', 'Bauhaus', 'Animalier', 'Muralism', 'Orientalism', 'Neo-Impressionism', 'Fauvism', "None"]

graph = Graph(f"bolt://localhost:{YOUR_PORT}",
              auth=("neo4j", YOUR_PASSWORD))
data = pl.from_dicts(graph.run("""
MATCH (artwork: Artwork)-->(artist: Artist)
OPTIONAL MATCH (artist)-->(movement: Movement)
WITH artwork, artist, movement
OPTIONAL MATCH (artwork)-->(medium: Medium)
WITH artwork, artist, movement, medium
RETURN artwork.id as id, artwork.name as artwork, artist.name as artist, movement.name as movement, medium.name as medium, artwork.image_url as url
""").data()) \
    .with_columns([
        pl.col("url").str.extract(r"([^/]*)$").alias("filename"),
        pl.col("url").str.extract(r"^(?:[^/]*/){4}([^/]*)").alias("catalog")
    ]) \
    .join(pl.read_parquet("./data/Artwork.parquet.gzip", columns=["id", "url"]).rename({"url": "source_url"}), on="id") \
    .to_pandas()
print(data.count())
# print(data["url"][0])

if not os.path.exists('original_images_movement'):
    os.mkdir('original_images_movement')
for movement_name in movement_names:
    if not os.path.exists(f'original_images_movement/{movement_name}'):
        os.makedirs(f'original_images_movement/{movement_name}')

for index, row in data.iterrows():
    try:
        movement = row['movement']
        # Split the URL into its components and quote each component this is needed only for generated images
        # scheme, netloc, path, query, fragment = urlsplit(
        #    row['image_url'])  # just url for generated files
        # path = quote(path)
        # query = quote(query, safe='=&')
        # fragment = quote(fragment)
        # urlunsplit((scheme, netloc, path, query, fragment))
        url = row['url']

        print(url)
        response = requests.get(url, verify=False)

        if response.status_code != 200:
            print(
                f'Error downloading image from {url}, response: {response.content}')
        else:
            # Get the name of the painting from the URL
            print("image located")
            painting_name = os.path.splitext(
                unquote(os.path.basename(url)))[0]

            with open(f'original_images_movement/{movement}/{painting_name}.jpg', 'wb') as f:
                f.write(response.content)
            # with open(f'generated_images/{painting_name}_{index}.jpg', 'wb') as f:
            #    f.write(response.content)#use this for generated images

    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
