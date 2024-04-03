from pathlib import Path
from typing import Callable
import threading

import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()

DATASET = pt.datasets.get_dataset("irds:beir/dbpedia-entity/dev")
IDX_PATH = Path("index").absolute()
if not (IDX_PATH / "data.properties").is_file():
    pt.index.IterDictIndexer(
        str(IDX_PATH),
        meta={
            "docno": 32,
            "text": 131072,
        },
    ).index(DATASET.get_corpus_iter())

BM25 = pt.BatchRetrieve(
    str(IDX_PATH),
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": ""},
    controls={"qe": "off"},
)


def search(query: str) -> pd.DataFrame:
    return (BM25 % 10).search(query)


def evaluate(df: pd.DataFrame, rewrite_func: Callable[[str], str] = None) -> float:
    if rewrite_func is None:
        pl = BM25
    else:
        pl = pt.apply.query(lambda q: rewrite_func(q["query"])) >> BM25
    return pt.Experiment(
        [pl],
        df,
        DATASET.get_qrels(),
        eval_metrics=["map"],
    )[
        "map"
    ][0]


def evaluate_all(rewrite_func: Callable[[str], str] = None) -> float:
    return evaluate(DATASET.get_topics().head(25), rewrite_func)


def evaluate_multi_thread(
    limit_df: pd.DataFrame,
    index: int,
    data: dict,
    rewrite_func: Callable[[str], str] = None,
) -> float:

    score = evaluate(limit_df, rewrite_func)

    data[index] = score


def evaluate_all_multi_thread(rewrite_func: Callable[[str], str] = None) -> float:
    topics = DATASET.get_topics()

    data = dict()
    threads = []

    for i in range(10):
        limit_df = topics.head(i).tail(18)

        t = threading.Thread(
            target=evaluate_multi_thread, args=(limit_df, i, data, rewrite_func)
        )
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    return data
