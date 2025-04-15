import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from FlagEmbedding import FlagReranker
import numpy as np
import json
import torch


if __name__ == "__main__":
    data_dir = "/processing_data/search/zengziyang/data/"
    model_dir = "/processing_data/search/zengziyang/models/"

    all_models = [
        "BAAI/bge-reranker-v2-m3",
    ]
    
    train_data_path = "exp3_qp_train.jsonl"
    test_data_path = "exp3_qp_test.jsonl"

    topk_list = [1, 5, 10, 20, 30, 50]
    batch_size = 8

    # load and process data
    query_answer_span_list, passage_list, query2passage = [], [], {}
    
    with open(train_data_path) as f:
        train_data = [json.loads(line) for line in f.readlines()]
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f.readlines()]
    
    for item in train_data:
        passage_list.append(item["content"])
        
    for item in test_data:
        query_answer_span_list.append(
            [item["question"], item["span_class"]]
        )
        passage_list.append(item["content"])
        query2passage[item["question"]] = item["content"]
            
    passage_list = list(set(passage_list))
    passage_list.sort()
    
    import random
    random.seed(42)
    random.shuffle(query_answer_span_list)
    before_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "before" in span][: 200]
    middle_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "middle" in span][: 200]
    after_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "after" in span][: 200]
    query_answer_span_list = before_query_answer_span_list + middle_query_answer_span_list + after_query_answer_span_list
    
    passage2id = {passage: idx for idx, passage in enumerate(passage_list)}
    labels = np.array(
        [[passage2id[query2passage[query]]] for query, _ in query_answer_span_list]
    )
    answer_span_list = [answer_span for _, answer_span in query_answer_span_list]
    print("number of all queries", len(query_answer_span_list))
    print("number of all passages", len(passage_list))
    print("min len of passage (words): ", min([len(passage.split(" ")) for passage in passage_list]))
    print("max len of passage (words): ", max([len(passage.split(" ")) for passage in passage_list]))
    
    for model_name in all_models:
        print(model_name)
        # load model
        model = FlagReranker(model_dir + model_name, use_fp16=True, max_length=8192)
        
        if "bge-reranker-v2-m3" in model_name:
            query_list = [item[0] for item in query_answer_span_list]
            pair_list = [[query, passage] for query in query_list for passage in passage_list]
            scores = model.compute_score(pair_list, normalize=False, batch_size=384)
            
        else:
            raise Exception(f"unsupported model {model_name}")

        # search topk
        # topk_index, _ = find_topk_by_vecs(q_vecs, p_vecs, max(topk_list))
        scores = scores.reshape(len(query_list), len(passage_list))
        sorted_indices = np.argsort(-scores, axis=1)
        topk_index = sorted_indices[:, :max(topk_list)]

        print(
            f"model, #queries, span_class, {', '.join([f'Recall@{k}' for k in topk_list])}"
        )
        # compute recall with different answer_start and top-k
        for span in ["before", "middle", "after"]:
            recall_at_k_list = []
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
            recall_at_k_list = [str(float(i)) for i in recall_at_k_list]  # for joining
            print(
                f"{model_name}, {len(selected_ids)}, {span}, {','.join(recall_at_k_list)}"
            )
            
        if model:
            del model
