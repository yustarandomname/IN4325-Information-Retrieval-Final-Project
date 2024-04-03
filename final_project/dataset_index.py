import pyterrier as pt
import os
import threading


def create_index(dataset: str, index_name: str, fields=["text"]):
    """
    Creates an index for a given dataset using the specified index name and fields.

    Parameters:
    - dataset: The dataset object containing the corpus to be indexed.
    - index_name: The name of the index to be created.
    - fields: A list of fields to be indexed. Default is ["text"].

    Returns:
    - index_ref: The reference to the created index.

    """
    indexer = pt.IterDictIndexer("./indices/" + index_name, verbose=False)
    index_ref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    return index_ref


def find_index(index_name: str):
    """
    Finds the reference to an existing index with the specified name.

    Parameters:
    - index_name: The name of the index to be found.

    Returns:
    - index_ref: The reference to the found index.

    """
    return pt.IndexRef.of("./indices/" + index_name)


def index(dataset: str, index_name: str, indeces: dict, lang: str, fields=["text"]):
    index_ref = create_index(dataset, index_name, fields)
    indeces[lang] = index_ref


def get_datasets_and_indeces(verbose=False):
    datasetNames: dict[str, str] = dict(
        {
            "fr": "wikir/fr14k",
            "es": "wikir/es13k",
            "en": "wikir/en1k",
            "it": "wikir/it16k",
        }
    )
    datasets: dict[str, pt.IndexRef] = dict()
    indeces: dict[str, pt.IndexRef] = dict()
    threads: list[threading.Thread] = []

    for [lang, datasetName] in datasetNames.items():
        datasetFolder = datasetName.replace("/", "_")
        dataset = pt.get_dataset("irds:" + datasetName)
        datasets[lang] = dataset

        if os.path.exists("./indices/" + datasetFolder + "/data.properties"):
            if verbose:
                print("Index", datasetFolder, "already exists")

            index_ref = find_index(datasetFolder)
            indeces[lang] = index_ref
        else:
            if verbose:
                print(
                    "Creating index",
                    datasetFolder,
                    " (takes around 1-3 minutes per dataset)",
                )

            thread = threading.Thread(
                target=index, args=(dataset, datasetFolder, indeces, lang)
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    return datasets, indeces
