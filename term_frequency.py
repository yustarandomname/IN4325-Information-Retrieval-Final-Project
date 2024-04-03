from collections import Counter
import math

# We have N documents per language and L langauges


# Convert each document to a set of words associated with its occurrence
def document_to_wordset(doc):
    # Create list of all words in doc
    allwords_list = doc.split(" ")

    # Length of this list is the total number of terms in doc
    total_terms = len(allwords_list)

    # Create a dictionary with terms as keys and occurrence count
    occurrence_set = Counter(allwords_list)

    return occurrence_set, total_terms


def compute_term_frequencies(doc_wordset: Counter, total_terms: int) -> Counter:
    result = doc_wordset.copy()
    # Simply divide the word occurrence in the wordset by the total number of terms in the document
    for term in result.keys():
        result[term] /= total_terms

    return result


def compute_idf(total_docs_in_corpus, docs_contain_term):
    return math.log((1 + total_docs_in_corpus) / (1 + docs_contain_term))


def compute_tfidf_per_term(frequency_wordsets: list[Counter]) -> Counter:
    # WARN: there are terms which do not occur in every document

    # Start by creating a counter that contains each term and the
    # number of documents that term appeared in
    document_occurrence_counter = Counter()

    for frequency_wordset in frequency_wordsets:
        for term in frequency_wordset.keys():
            # If first encounter; set to one
            if term not in document_occurrence_counter.keys():
                document_occurrence_counter[term] = 1
            # Otherwise add one encounter
            else:
                document_occurrence_counter[term] += 1

    # For each term, compute the average term frequency,
    # using the document count in aggregate_counter
    avg_term_frequency_counter = Counter()

    for term in document_occurrence_counter.keys():

        avg_term_frequency_counter[term] = 0

        for frequency_wordset in frequency_wordsets:
            if term in frequency_wordset.keys():
                avg_term_frequency_counter[term] += frequency_wordset[term]

        avg_term_frequency_counter[term] /= document_occurrence_counter[term]

    # For each term, compute the inverse document frequency for the documents
    # in the given list
    term_inverse_document_frequency_counter = Counter()

    # The total number of documents in the corpus is simply in this case
    total_documents = len(frequency_wordsets)

    # For each term
    for term in document_occurrence_counter.keys():
        term_inverse_document_frequency_counter[term] = compute_idf(
            total_documents, document_occurrence_counter[term]
        )

    # Finally, combine tf-idf for each term using the above two created counters
    tfidf_counter = Counter()

    for term in document_occurrence_counter.keys():
        tfidf_counter[term] = (
            avg_term_frequency_counter[term]
            * term_inverse_document_frequency_counter[term]
        )

    return tfidf_counter


def get_most_common_terms(documents_translated: list[str], amount=100):
    all_frequency_wordsets: list[Counter] = []

    for doc in documents_translated:
        doc_wordset, total_terms = document_to_wordset(doc)

        frequency_wordset = compute_term_frequencies(doc_wordset, total_terms)
        all_frequency_wordsets.append(frequency_wordset)

    tfidf_per_term = compute_tfidf_per_term(all_frequency_wordsets)

    # Based on tfidf, extract the 5 'most domain-specific' terms
    domain_specific_terms = [item[0] for item in tfidf_per_term.most_common(amount)]

    print(domain_specific_terms)
    return domain_specific_terms
