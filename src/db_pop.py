import polars as pl
import os

from functools import partial
from tempfile import NamedTemporaryFile
from py2neo import Graph

YOUR_PASSWORD = "password"
YOUR_PORT = 7687

# does not work, issues with Neo4j


def parquet_to_neo4j(path: str, query: str, graph: Graph):
    with NamedTemporaryFile(suffix=".csv", delete=False) as file:
        pl.read_parquet(path).write_csv(file.name)
        combined_query = query % f"file:///{file.name}"
        graph.run(combined_query)


if __name__ == "__main__":
    graph = Graph(f"bolt://localhost:{YOUR_PORT}",
                  auth=("neo4j", YOUR_PASSWORD))
    graph.delete_all()
    # graph.run('match (n) detach delete n') # alternative

    try:
        indexes = graph.run('show indexes yield name').to_data_frame()[
            'name']  # drops all indices
        for index in indexes:
            graph.run(f'drop index {index}')
    except:
        pass

    # fn = partial(parquet_to_neo4j, graph = graph)

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Specialization.csv" AS csvLine
    MERGE (s:Specialization {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        description: csvLine.description
        })
    """
    )

    graph.run('CREATE INDEX specialization FOR (n:Specialization) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Movement.csv" AS csvLine
    MERGE (m:Movement {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        description: csvLine.description
        })
    """
    )

    graph.run('CREATE INDEX movement FOR (n:Movement) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Academy.csv" AS csvLine
    MERGE (a:Academy {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        description: csvLine.description
        })
    """
    )

    graph.run('CREATE INDEX academy FOR (n:Academy) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Medium.csv" AS csvLine
    MERGE (m:Medium {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        description: csvLine.description
        })
    """
    )

    graph.run('CREATE INDEX medium FOR (n:Medium) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Places.csv" AS csvLine
    MERGE (m:Place {
        id: toInteger(csvLine.id), 
        name: csvLine.name
        })
    """
    )

    graph.run('CREATE INDEX place FOR (n:Place) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Places.csv" AS csvLine
    MATCH (p1:Place {id: toInteger(csvLine.id)}), (p2:Place {id: toInteger(csvLine.parent)})
    MERGE (p1) -[r:LOCATED_IN]-> (p2)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/ArtistPicture.csv" AS csvLine
    CREATE (m:Picture {
        id: toInteger(csvLine.id), 
        url: csvLine.url,
        source_url: csvLine.source_url,
        caption: csvLine.caption
        })
    """
    )

    graph.run('CREATE INDEX picture FOR (n:Picture) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Generated.csv" AS csvLine
    CREATE (m:Generated {
        url: csvLine.url
        })
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.csv" AS csvLine
    CREATE (m:Artwork {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        image_url: csvLine.image_url,
        rating: toInteger(csvLine.rating),
        summary: csvLine.summary,
        year: toIntegerOrNull(csvLine.year),
        location: csvLine.location
        })
    """
    )

    graph.run('CREATE INDEX artwork FOR (n:Artwork) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.csv" AS csvLine
    MATCH (a:Artwork {id: toInteger(csvLine.id)}), (m:Medium {id: toIntegerOrNull(csvLine.medium)})
    MERGE (a) -[r:USES]-> (m)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Generated.csv" AS csvLine
    MATCH (a:Artwork {id: toInteger(csvLine.source_artwork)}), (g:Generated {url: csvLine.url})
    MERGE (g) -[r:BASED_ON]-> (a)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Recommendation.csv" AS csvLine
    MATCH (a:Artwork {id: toInteger(csvLine.artwork)}), (recommendation:Artwork {id: toInteger(csvLine.recommended)})
    MERGE (a) -[r:RECOMMENDS]-> (recommendation)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.csv" AS csvLine
    CREATE (m:Artist {
        id: toInteger(csvLine.id), 
        name: csvLine.name,
        url: csvLine.url,
        summary: csvLine.summary
        })
    """
    )

    graph.run('CREATE INDEX artist FOR (n:Artist) ON (n.id)')

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.csv" AS csvLine
    MATCH (a:Artist {id: toInteger(csvLine.id)}), (picture:Picture {id: toIntegerOrNull(csvLine.picture)})
    MERGE (a) -[r:IMAGE]-> (picture)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.csv" AS csvLine
    MATCH (a:Artist {id: toInteger(csvLine.id)}), (birthplace:Place {id: toIntegerOrNull(csvLine.birthplace)})
    MERGE (a) -[r:BORN_IN]-> (birthplace)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artist.csv" AS csvLine
    MATCH (a:Artist {id: toInteger(csvLine.id)}), (deathplace:Place {id: toIntegerOrNull(csvLine.deathplace)})
    MERGE (a) -[r:DIED_IN]-> (deathplace)
    SET r.cause = csvLine.cause_of_death
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Apprenticeship.csv" AS csvLine
    MATCH (student:Artist {id: toInteger(csvLine.student_id)}), (teacher:Artist {id: toIntegerOrNull(csvLine.teacher_id)})
    MERGE (student) -[r:APPRENTICE_OF]-> (teacher)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/Artwork.csv" AS csvLine
    MATCH (artwork:Artwork {id: toInteger(csvLine.id)}), (artist:Artist {id: toIntegerOrNull(csvLine.artist)})
    MERGE (artwork) -[r:MADE_BY]-> (artist)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/ArtistSpecializations.csv" AS csvLine
    MATCH (s:Specialization {id: toInteger(csvLine.specialty_id)}), (artist:Artist {id: toIntegerOrNull(csvLine.artist_id)})
    MERGE (s) <-[r:SPECIALIZED_IN]- (artist)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/ArtistMovements.csv" AS csvLine
    MATCH (s:Movement {id: toInteger(csvLine.movement_id)}), (artist:Artist {id: toIntegerOrNull(csvLine.artist_id)})
    MERGE (s) <-[r:BELONGS_TO]- (artist)
    """
    )

    graph.run(
        """
    LOAD CSV WITH HEADERS FROM "https://kuleuven-datathon-2023.s3.eu-central-1.amazonaws.com/data/ArtistEducation.csv" AS csvLine
    MATCH (s:Academy {id: toInteger(csvLine.academy_id)}), (artist:Artist {id: toIntegerOrNull(csvLine.artist_id)})
    MERGE (s) <-[r:EDUCATED_AT]- (artist)
    """
    )
    
