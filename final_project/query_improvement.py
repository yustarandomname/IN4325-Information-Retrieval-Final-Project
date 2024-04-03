import re

import pandas as pd
import pyterrier as pt
import unidecode
from deep_translator import single_detection
from dotenv import dotenv_values
from dataset_index import get_datasets_and_indeces
from term_frequency import get_most_common_terms
from translation import document_translation, traslate_queries


def sanitise_query(query: str):
    """
    Sanitises a query by removing special characters and converting it to lowercase.

    Parameters:
    - query: The query to be sanitised.

    Returns:
    - sanitised_query: The sanitised query.

    """
    decoded_query = unidecode.unidecode(query)
    sanitised_query = re.sub(r"[^a-zA-Z0-9 ]", "", decoded_query)
    return sanitised_query.lower()


def retrieve_documents(
    languages: list[str],
    indeces: dict[str, pt.IndexRef],
    datasets: dict,
    qs: dict[str, str],
    NUM_RESULTS=10,
):
    document_data: dict[str, pd.DataFrame] = dict()

    for lang in languages:
        index_ref = indeces[lang]
        dataset = datasets[lang]

        pipeline = pt.BatchRetrieve(
            index_ref, wmodel="BM25", metadata=["docno"], num_results=NUM_RESULTS
        ) >> pt.text.get_text(dataset, "text")

        sanitised_query = sanitise_query(qs[lang])

        pandas_df: pd.DataFrame = pipeline.search(sanitised_query)
        document_data[lang] = pandas_df

    return document_data


def query_improvement(q0, NUM_RESULTS=10, TERM_AMOUNT=10):

    env = dotenv_values("../.env")  # replace ".env.example" with .env file path

    if env["LANGUAGE_DETECT_API_KEY"] == "YOUR API KEY":
        raise Exception(
            "Please replace 'YOUR API KEY' with your actual API key in the .env file"
        )

    detection_api_key = env["LANGUAGE_DETECT_API_KEY"]

    if not pt.started():
        pt.init()

    languages = ["en", "fr", "it", "es"]

    l0 = single_detection(q0, api_key=detection_api_key)

    qs = traslate_queries(q0, l0, languages)
    # print(qs)

    datasets, indeces = get_datasets_and_indeces()
    # print(datasets, indeces)

    document_data = retrieve_documents(languages, indeces, datasets, qs, NUM_RESULTS)
    # print(document_data)

    documents_translated = document_translation(document_data)
    # print(documents_translated)

    domain_specific_terms = get_most_common_terms(documents_translated, TERM_AMOUNT)
    # print(domain_specific_terms)
    newQuery = q0 + " " + " ".join(domain_specific_terms)

    print(q0, "->", domain_specific_terms)
    return newQuery
