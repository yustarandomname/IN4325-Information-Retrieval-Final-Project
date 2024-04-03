import threading
from deep_translator import GoogleTranslator
from pandas import DataFrame


def translate(lang: str, l0: str, q0: str, qs: dict):
    gt = GoogleTranslator(source=l0, target=lang)
    translated = gt.translate(text=q0)
    qs[lang] = translated


def traslate_queries(q0, l0, languages):
    threads = []
    qs = dict({l0: q0})

    for lang in languages:
        if lang != l0:
            t1 = threading.Thread(target=translate, args=(lang, l0, q0, qs))
            t1.start()
            threads.append(t1)

    for t in threads:
        t.join()

    return qs


def translate_document(
    lang: str,
    index: int,
    document_text: str,
    documents_translated_dict: dict[str, list[str]],
):
    gt = GoogleTranslator(source=lang, target="en")
    translated = gt.translate(text=document_text)

    documents_translated_dict[lang][index] = translated


def document_translation(document_data: dict[str, DataFrame]):
    documents_translated_dict: dict[str, list[str]] = {}

    threads = []

    for lang, docs in document_data.items():
        text_docs: list[str] = docs["text"].tolist()
        documents_translated_dict[lang] = [None] * len(text_docs)

        for index, text in enumerate(text_docs):
            t = threading.Thread(
                target=translate_document,
                args=(lang, index, text, documents_translated_dict),
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    # Join the translated documents to one list
    documents_translated: list[str] = []
    for lang, docs in documents_translated_dict.items():
        documents_translated += docs

    return documents_translated
