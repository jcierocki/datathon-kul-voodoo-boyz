import polars as pl
import requests

from pathlib import Path

FILENAME_LEN_WORDS = 8
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}


def download_images(urls: list[str], filenames: list[str], destination: str, ca_cert_download_path: str = "https://mkcert.org/generate/") -> None:
    if len(urls) != len(filenames):
        raise Exception("You need to provide the same number of urls and filenames.")

    ca_file = Path("certs.pem")
    try:
        ca_cert = requests.get(ca_cert_download_path, headers=HEADERS).content
    except Exception:
        raise Exception("Unable to download CA cert")
    
    with open(str(ca_file), "wb") as f:
        f.write(ca_cert)

    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)

    for i, url, filename in zip(range(len(urls)), urls, filenames):
        try:
            response = requests.get(url, headers=HEADERS, verify=str(ca_file))
            if response.status_code != 200:
                print(f'Error downloading image from {url}, response: {response.content}')
            else:
                with open(f"{destination}/{filename}_{i}.jpg", 'wb') as f:
                    f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')


    ca_file.unlink()


if __name__ == "__main__":
    pl.Config.set_fmt_str_lengths(150)

    artwork_filename_dictionary_df = pl.read_parquet("data/Artwork.parquet.gzip", columns=["image_url", "name"])\
        .with_columns([
            pl.col("name").str.replace_all(r"[^\P{P}-]+", "").apply(lambda s: "_".join(s.lower().split(" ")[:FILENAME_LEN_WORDS])).alias("filename")
        ]) \
        .rename({"image_url": "url"})
        
    print(artwork_filename_dictionary_df.head())

    generated_filename_dictionary_df = pl.read_parquet("data/Generated.parquet.gzip", columns=["url"]).with_columns([
        pl.col("url").str.extract(r"([^/]*)$").str.replace_all(r"[^\P{P}-]+", "").apply(lambda s: "_".join(s.lower().split(" ")[:FILENAME_LEN_WORDS])).alias("filename"),
        pl.col("url").str.replace_all(r" ", "+")
    ])
    
    print(generated_filename_dictionary_df.head())
    
    # urls_custom = urls_original.filter(pl.col("rowid") == 5371).get_column("image_url").to_list()
    # print(urls_custom[0])
    download_images(
        artwork_filename_dictionary_df.get_column("url").to_list(),
        artwork_filename_dictionary_df.get_column("filename").to_list(), 
        "data/images/original"
    )

    download_images(
        generated_filename_dictionary_df.get_column("url").to_list(),
        generated_filename_dictionary_df.get_column("filename").to_list(), 
        "data/images/generated"
    )