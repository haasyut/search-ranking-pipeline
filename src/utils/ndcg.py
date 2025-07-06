import numpy as np
import collections

def validate(qids, targets, preds, k):
    """
    Evaluate model ranking quality using NDCG@k.

    Parameters
    ----------
    qids : array-like
        Query IDs corresponding to each document.
    targets : array-like
        Ground truth relevance scores for each document.
    preds : array-like
        Predicted relevance scores for each document.
    k : int
        The cutoff rank for computing NDCG@k.

    Returns
    -------
    average_ndcg : float
        Average NDCG@k over all queries.
    every_qid_ndcg : dict
        Per-query NDCG@k values (ordered by qid).
    """
    query_groups = get_groups(qids)  # (qid, start_idx, end_idx)
    all_ndcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1]
        t_results = targets[a:b][predicted_sorted_indexes]

        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = dcg_val / idcg_val if idcg_val != 0 else 0.0
        all_ndcg.append(ndcg_val)
        every_qid_ndcg[qid] = ndcg_val

    average_ndcg = np.nanmean(all_ndcg)
    return average_ndcg, every_qid_ndcg


def get_groups(qids):
    """
    Yield group boundaries for each query id in the list.

    Parameters
    ----------
    qids : array-like
        A list of query ids for each document.

    Yields
    ------
    tuple
        (query_id, start_index, end_index)
    """
    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)


def group_queries(training_data, qid_index):
    """
    Group training data by query id.

    Parameters
    ----------
    training_data : array-like
        Each row is [relevance score, qid, features...]
    qid_index : int
        Index where query id is stored in each row.

    Returns
    -------
    dict
        Dictionary: {qid: list of row indices}
    """
    query_indexes = {}
    for index, record in enumerate(training_data):
        query_indexes.setdefault(record[qid_index], []).append(index)
    return query_indexes


def dcg_k(scores, k):
    """
    Compute Discounted Cumulative Gain (DCG) at rank k.

    Parameters
    ----------
    scores : list
        Relevance scores in predicted ranking order.
    k : int
        Truncation rank.

    Returns
    -------
    float
        DCG@k
    """
    return np.sum([
        (2 ** scores[i] - 1) / np.log2(i + 2)
        for i in range(min(len(scores), k))
    ])


def ideal_dcg_k(scores, k):
    """
    Compute Ideal DCG@k — i.e., DCG for perfect ranking.

    Parameters
    ----------
    scores : list
        Relevance scores.
    k : int
        Truncation rank.

    Returns
    -------
    float
        Ideal DCG@k
    """
    sorted_scores = sorted(scores, reverse=True)
    return dcg_k(sorted_scores, k)
