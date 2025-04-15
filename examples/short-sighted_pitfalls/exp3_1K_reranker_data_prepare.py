import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
import commercial_encoder_api
import json
import pickle
import torch


def find_topk_by_vecs(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)

    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


if __name__ == "__main__":
    data_dir = "/processing_data/search/zengziyang/data/"
    model_dir = "/processing_data/search/zengziyang/models/"
    
    train_data_path = "exp3_qp_train.jsonl"
    test_data_path = "exp3_qp_test.jsonl"

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
    before_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "before" in span][: 500]
    middle_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "middle" in span][: 500]
    after_query_answer_span_list = [[query, span] for query, span in query_answer_span_list if "after" in span][: 500]
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
    
    model_name = "nvidia/NV-Embed-v2"
    print(model_name)
    # load model
    model = SentenceTransformer(
        model_dir +  model_name, trust_remote_code=True
    )
    
    # Each query needs to be accompanied by an corresponding instruction describing the task.
    task_name_to_instruct = {
        "example": "Given a question, retrieve passages that answer the question",
    }
    query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "

    def add_eos(input_examples):
        input_examples = [
            input_example + model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    model.max_seq_length = 3072
    # model.max_seq_length = 32768
    model.tokenizer.padding_side = "right"
    q_vecs = model.encode(
        add_eos([item[0] for item in query_answer_span_list]),
        show_progress_bar=True,
        batch_size=256,
        normalize_embeddings=True,
        prompt=query_prefix,
    )
    p_vecs = model.encode(
        add_eos(passage_list),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )

    # search topk
    topk_list = [1, 5, 10, 20, 50, 100, 200, 300]
    topk_index, _ = find_topk_by_vecs(q_vecs, p_vecs, max(topk_list))

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
        
    # topk_index [query num，TopK]
    with open('exp3_1K_reranker_topK_index.pkl', 'wb') as f:
        pickle.dump(topk_index, f)
    
    print(topk_index.shape)
    
    
    