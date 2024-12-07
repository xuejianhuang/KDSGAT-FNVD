import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import json
# DBpedia Spotlight API URL
DBPEDIA_SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en/annotate"
HEADERS = {'accept': 'application/json'}


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def escape_entity(entity):
    """Escape special characters in the entity name for SPARQL query."""
    return entity.replace("(", "\(").replace(")", "\)").replace("'", "\\'").replace(".", "\.").replace(",","\,").replace("!","\!").replace("&","\&").replace("%","\%")

def disambiguate_entity(text, confidence=0.5):
    """Disambiguate entities in the given text using DBpedia Spotlight."""
    params = {'text': text, 'confidence': confidence}
    try:
        response = requests.get(DBPEDIA_SPOTLIGHT_URL, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        result = response.json()

        entities = [
            (entity['@surfaceForm'].lower(), entity['@URI'].split('/')[-1])
            for entity in result.get('Resources', [])
        ]

        return zip(*entities) if entities else ([], [])
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return [], []


def get_entity_concepts(entity_names):
    """Retrieve DBpedia concepts associated with a list of entity names."""
    if entity_names is None or len(entity_names)==0:
        return {}
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    # Escape special characters in entity names
    formatted_entities = " ".join(f"dbr:{escape_entity(entity)}" for entity in entity_names)

    query = f"""
    SELECT ?entity ?concept WHERE {{
        VALUES ?entity {{ {formatted_entities} }}
        ?entity rdf:type ?concept .
        FILTER(STRSTARTS(STR(?concept), "http://dbpedia.org/ontology/"))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        entity_concepts = {}
        for result in results["results"]["bindings"]:
            entity = result["entity"]["value"].split('/')[-1]
            concept = result["concept"]["value"].split('/')[-1]
            entity_concepts.setdefault(entity, []).append(concept)
        return entity_concepts
    except Exception as e:
        print(f"Error querying SPARQL endpoint: {e}")
        print(query)
        return {}