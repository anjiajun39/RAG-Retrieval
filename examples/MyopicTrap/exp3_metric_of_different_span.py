import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import json
from utils import (
    find_topk_by_single_vecs,
    find_topk_by_multi_vecs,
    find_topk_by_reranker,
)
from sklearn.metrics import ndcg_score

if __name__ == "__main__":
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Myopic Trap Exp3")
    parser.add_argument("--data_name_or_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_type", default="local", type=str, choices=["local", "api"])
    parser.add_argument("--first_stage_model_name_or_path", type=str, default=None)
    parser.add_argument("--first_stage_model_type", type=str, choices=["local", "api"])
    parser.add_argument("--cache_path", type=str, default="./rerank_cache.pickle", help="Path to save the first stage cache for reranking.")
    parser.add_argument(
        "--reranker_sampling",
        action="store_true",
        help="Whether to sample the queries for reranker. Only for reranker.",
    )
    parser.add_argument(
        "--score_type",
        required=True,
        type=str,
        choices=["single_vec", "multi_vec", "reranker"],
    )
    args = parser.parse_args()

    data_name_or_path = args.data_name_or_path
    model_name_or_path = args.model_name_or_path
    model_type = args.model_type
    topk_list = [5, 10, 20, 30, 50, 100]

    # load and process data
    query_answer_span_list, passage_list, query2passage = [], [], {}

    with open(data_name_or_path) as f:
        all_data = [json.loads(line) for line in f.readlines()]

    for item in all_data:
        query_answer_span_list.append([item["question"], item["span_class"]])
        passage_list.append(item["content"])
        query2passage[item["question"]] = item["content"]

    passage_list = list(set(passage_list))
    passage_list.sort()
    passage2id = {passage: idx for idx, passage in enumerate(passage_list)}
    
    
    if args.score_type == "reranker" and args.reranker_sampling:
        print("Sampling 1w query for Reranker due to efficiency")
        import random
        random.seed(42)
        random.shuffle(query_answer_span_list)
        before_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "before" in span][: 3300]
        middle_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "middle" in span][: 3300]
        after_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "after" in span][: 3300]
        query_answer_span_list = before_query_answer_span_list + middle_query_answer_span_list + after_query_answer_span_list
    else:
        print("Using all queries for Reranker")

    query_list = [item[0] for item in query_answer_span_list]
    labels = np.array(
        [[passage2id[query2passage[query]]] for query, _ in query_answer_span_list]
    )
    answer_span_list = [answer_span for _, answer_span in query_answer_span_list]
    print("Data Statistics:")
    print(f"data_name_or_path: {data_name_or_path}")
    print("number of all queries", len(query_list))
    print(
        "min len of query (words): ",
        min([len(query.split(" ")) for query in query_list]),
    )
    print(
        "max len of query (words): ",
        max([len(query.split(" ")) for query in query_list]),
    )
    print("number of all passages", len(passage_list))
    print(
        "min len of passage (words): ",
        min([len(passage.split(" ")) for passage in passage_list]),
    )
    print(
        "max len of passage (words): ",
        max([len(passage.split(" ")) for passage in passage_list]),
    )

    print("Searching Topk ...")
    print(f"Using: {model_name_or_path} {model_type} {args.score_type}")
    if args.score_type == "single_vec":
        topk_index, topk_scores = find_topk_by_single_vecs(
            model_name_or_path, model_type, query_list, passage_list, max(topk_list)
        )
    elif args.score_type == "multi_vec":
        topk_index, topk_scores = find_topk_by_multi_vecs(
            model_name_or_path, model_type, query_list, passage_list, max(topk_list)
        )
    elif args.score_type == "reranker":
        topk_index, topk_scores = find_topk_by_reranker(
            reranker_model_name_or_path=model_name_or_path,
            embedding_model_name_or_path=args.first_stage_model_name_or_path,
            reranker_model_type=args.model_type,
            embedding_model_type=args.first_stage_model_type,
            query_list=query_list,
            passage_list=passage_list,
            topk=max(topk_list),
            recall_topk=max(topk_list),
            cache_path=args.cache_path,
        )
    print("Search Topk Done.")
    print(f"Result shape: {topk_scores.shape}")

    print("--------Evaluation--------")
    print(
        f"model, #queries, span_class, {', '.join([f'Recall@{k}' for k in topk_list])}, {', '.join([f'NDCG@{k}' for k in topk_list])}"
    )
    # compute recall with different answer_start and top-k
    for span in ["before", "middle", "after"]:
        recall_at_k_list = []
        ndcg_at_k_list = []
        selected_ids = [
            idx
            for idx, answer_span in enumerate(answer_span_list)
            if span in answer_span
        ]
        for topk in topk_list:
            recall_at_k_list.append(
                (topk_index[selected_ids, :topk] == labels[selected_ids, :]).sum()
                / len(selected_ids)
            )
            ndcg_at_k_list.append(
                ndcg_score(
                    y_true=topk_index[selected_ids, :] == labels[selected_ids, :],
                    y_score=topk_scores[selected_ids, :],
                    k=topk,
                )
            )

        recall_at_k_list = [str(float(i)) for i in recall_at_k_list]  # for joining
        ndcg_at_k_list = [str(float(i)) for i in ndcg_at_k_list]  # for joining
        print(
            f"{model_name_or_path}, {len(selected_ids)}, {span}, {','.join(recall_at_k_list)}, {','.join(ndcg_at_k_list)}"
        )
    print("-------------------------------")
