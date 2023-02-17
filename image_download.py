import pandas
import pyarrow
import os
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

if not os.path.exists('original_images'):
    os.mkdir('original_images')


for index, row in data.iterrows():
    try:
        # Split the URL into its components and quote each component
        scheme, netloc, path, query, fragment = urlsplit(
            row['image_url'])  # just url for generated files
        path = quote(path)
        query = quote(query, safe='=&')
        fragment = quote(fragment)
        url = urlunsplit((scheme, netloc, path, query, fragment))

        response = requests.get(url, headers=headers, verify=False)
        if response.status_code != 200:
            print(
                f'Error downloading image from {url}, response: {response.content}')
        else:
            # Get the name of the painting from the URL
            painting_name = os.path.splitext(
                unquote(os.path.basename(url)))[0]

            with open(f'generated_images/{painting_name}_{index}.jpg', 'wb') as f:
                f.write(response.content)

    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
