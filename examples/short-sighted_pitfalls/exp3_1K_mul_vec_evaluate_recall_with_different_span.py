import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import faiss
import commercial_encoder_api
import json
import torch
import os
import numpy as np
from tqdm import tqdm

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel


def find_topk_by_vecs(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)

    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


if __name__ == "__main__":
    data_dir = "/processing_data/search/zengziyang/data/"
    model_dir = "/processing_data/search/zengziyang/models/"

    all_models = [
        "infgrad/dewey_en_beta",
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
        if "infgrad/dewey_en_beta" in model_name:
            class TextSpan(BaseModel):
                s: int
                e: int
                text: Optional[str] = None
                module_name: str = "text_span"      
        
            model = AutoModel.from_pretrained(
                model_dir + "infgrad/dewey_en_beta",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).cuda().bfloat16()
            model.tokenizer = AutoTokenizer.from_pretrained(model_dir + "infgrad/dewey_en_beta")
        
        if "infgrad/dewey_en_beta" in model_name:
            model.max_seq_length = 32 * 1024
            RETRIEVE_Q_PROMPT = "<|START_INSTRUCTION|>Answer the question<|END_INSTRUCTION|>"
            RETRIEVE_P_PROMPT = "<|START_INSTRUCTION|>Candidate document<|END_INSTRUCTION|>"
            
            query_vectors, _ = model.encode(
                sentences=[item[0] for item in query_answer_span_list],
                use_cuda=True,
                show_progress_bar=True,
                chunk_size=-1,
                chunk_overlap=32,
                convert_to_tensor=False,
                max_seq_length=32 * 1024,
                batch_size=8 * 300,
                normalize_embeddings=True,
                prompt=RETRIEVE_Q_PROMPT,
                fast_chunk=False
            )
            q_vecs = np.array([vecs[1, :] for vecs in query_vectors])
            
            # spans_list contail each chunk's span, you can use span to get text
            spans_list: List[List[TextSpan]]
            passage_vectors_list: List[np.ndarray]
            passage_vectors_list, spans_list = model.encode(
                sentences=passage_list,
                use_cuda=True,
                show_progress_bar=True,
                chunk_size=256,
                chunk_overlap=64,
                convert_to_tensor=False,
                max_seq_length=32 * 1024,
                batch_size=8 * 60,
                normalize_embeddings=True,
                prompt=RETRIEVE_P_PROMPT,
                fast_chunk=True,  # if fast_chunk is true, directly chunk on input ids, else using RecursiveCharacterTextSplitter
            )             
            cnt = 0
            chunk_id2passage_id = {}
            for passage_id, chunks in enumerate(passage_vectors_list):
                for chunk_id, chunk in enumerate(chunks):
                    chunk_id2passage_id[cnt] = passage_id
                    cnt += 1
            
            p_vecs = np.concatenate(passage_vectors_list, axis=0)  # shape 是 (n * m, 2048)
            
        else:
            raise Exception(f"unsupported model {model_name}")
        
        topk_index, topk_scores = find_topk_by_vecs(q_vecs, p_vecs, max(topk_list) * 100)
        new_topk_index = []
        new_topk_scores = []
        for chunk_ids, chunk_scores in tqdm(zip(topk_index, topk_scores), desc="modify topk_index and topk_scores", disable=True):
            # processed by row
            row_ids, row_scores, passage_id_set = [], [], set()
            for idx, chunk_id in enumerate(chunk_ids):
                passage_id = chunk_id2passage_id[chunk_id]
                if passage_id not in passage_id_set:
                    passage_id_set.add(passage_id)
                    row_ids.append(passage_id)
                    row_scores.append(chunk_scores[idx])
            new_topk_index.append(row_ids[:max(topk_list)])
            new_topk_scores.append(row_scores[:max(topk_list)])
            
        topk_index = np.array(new_topk_index)
        topk_scores = np.array(new_topk_scores)
        topk_index, topk_scores = topk_index[:, :max(topk_list)], topk_scores[:, :max(topk_list)]   
        

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
