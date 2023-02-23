from cmath import nan
import pandas
import pyarrow
import os
import math
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote, urlsplit, urlunsplit, unquote

# data = pandas.read_parquet("./data/Generated.parquet.gzip")
data = pandas.read_parquet("./data/Artwork.parquet.gzip")
print(data.count())  # example of operation on the returned DataFrame
ca_bundle = "/usr/lib/ssl/cert.pem"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

# if not os.path.exists('generated_images'):
#    os.mkdir('generated_images')
medium_names = ["Oil_on_canvas", "Oil_on_panel", "Mixed_media", "Oil_on_wood", "Marble", "Oil_on_paper",
                "Tempera_on_panel", "Oil_on_masonite", "Fresco", "Bronze", "Paper", "Tempera_on_canvas", "Oil_on_copper"]


def map_medium(value):
    # convert value between 0 and 12 to an index
    medium_index = int(value * (len(medium_names)-1) / 12.0)
    return medium_names[medium_index]


if not os.path.exists('original_images'):
    os.mkdir('original_images')
for medium_name in medium_names:
    if not os.path.exists(f'original_images/{medium_name}'):
        os.makedirs(f'original_images/{medium_name}')


for index, row in data.iterrows():
    try:
        if 'medium' in data.columns:
            medium = row['medium']
            if math.isnan(medium):
                continue
            else:
                medium = map_medium(medium)
        # Split the URL into its components and quote each component this is needed only for generated images
        # scheme, netloc, path, query, fragment = urlsplit(
        #    row['image_url'])  # just url for generated files
        # path = quote(path)
        # query = quote(query, safe='=&')
        # fragment = quote(fragment)
        # urlunsplit((scheme, netloc, path, query, fragment))
        url = row['image_url']

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

            with open(f'original_images/{medium}/{painting_name}_{index}.jpg', 'wb') as f:
                f.write(response.content)
            # with open(f'generated_images/{painting_name}_{index}.jpg', 'wb') as f:
            #    f.write(response.content)#use this for generated images

    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
