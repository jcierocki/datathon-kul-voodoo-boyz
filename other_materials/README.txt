############
Data Summary
############
The data for the KU Leuven Datathon 2023 consists of several tables related to artworks, artists, movements, academies, specializations, medium, and places. The tables contain information such as the name, description, image URL, and relationships between different concepts.

In addition to the main data tables, there are also AI generated images created using Text-To-Image technology.

The data is available in three different ways:

1. As a zip file stored in CVS format.
2. As a zip file stored in parquet.gzip format.
3. Hosted online at the URL https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/{file_name}.{file_extension}.

A jupyter notebook file is also included in the ZIP file that provides additional information and insights into the data.

##############
Table Overview
##############
Artworks Table: contains information about individual artworks, including their name, URL, image URL, artist ID, rating, summary, year, medium, and location. Artworks can be linked to each other based on the Recommendation table.

Artists Table: contains information about individual artists, including their name, URL, image URL, summary, birth- and deathplace, birth- and deathdate, and cause of death. Artists can also be linked to each other through apprenticeships.

Movements Table: gives a unique ID to certain art movements and contains the movement name, ID, and description.

Academies Table: gives a unique ID to academies attended by artists and contains the academy name, ID, and description.

Specializations Table: gives a unique ID to specializations of artists and contains the specialization name, ID, and description.

Medium Table: gives a unique ID to the medium used in artworks and contains the medium name, ID, and description.

Places Table: gives a unique ID to the birth- and deathplaces of artists and is hierarchically organized with each place having an ID, Name, and parent.

AI generated images using Text-To-Speech technology: contains images generated using Text-To-Speech technology.

We hope that this data will provide valuable insights and inspiration for your projects. Good luck!