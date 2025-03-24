import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
import commercial_encoder_api
import json
import torch


def find_topk_by_vecs(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)

    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


if __name__ == "__main__":
    data_cache_dir = ""
    model_cache_dir = ""

    base_dir = ""
    
    all_models = [
        "BAAI/bge-m3",
        "NovaSearch/jasper_en_vision_language_v1",
        "nvidia/NV-Embed-v2",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "jinaai/jina-embeddings-v2-base-en",
        "jinaai/jina-embeddings-v3",
        "infgrad/very_awesome",
        "api_voyage",
        "api_openai",
    ]
    # model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # model_name = "api_voyage" # "api_openai", "api_cohere", "api_voyage", "api_jina"
    
    train_data_path = "examples/short-sighted_pitfalls/exp3_qp_train.jsonl"
    test_data_path = "examples/short-sighted_pitfalls/exp3_qp_train.jsonl"

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
        if "api" not in model_name:
            if "infgrad/very_awesome" in model_name:
                model = SentenceTransformer(
                    base_dir +  model_name,
                    trust_remote_code=True,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,  # fp16 容易计算出nan
                        "attn_implementation": "flash_attention_2"
                    },
                    config_kwargs={"single_vector_type": "cls_add_mean"} # mean, cls, cls_add_mean
                ).cuda().bfloat16().eval()
            else:
                model = SentenceTransformer(
                    base_dir +  model_name, trust_remote_code=True, cache_folder=model_cache_dir
                )
        else:
            model = None
        
        # get q,p vecs
        if "jina-embeddings-v3" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                task="retrieval.query",
                prompt_name="retrieval.query",
                show_progress_bar=True,
                batch_size=batch_size * 3,
                normalize_embeddings=True,
            )
            p_vecs = model.encode(
                passage_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "bge-m3" in model_name or "jina-embeddings-v2-base-en" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                show_progress_bar=True,
                batch_size=batch_size * 3,
                normalize_embeddings=True,
            )
            p_vecs = model.encode(
                passage_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "jasper_en_vision_language_v1" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                show_progress_bar=True,
                prompt_name="s2p_query",
                batch_size=batch_size * 3,
                normalize_embeddings=True,
            )
            p_vecs = model.encode(
                passage_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "nvidia" in model_name:
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

            model.max_seq_length = 32768
            model.tokenizer.padding_side = "right"
            q_vecs = model.encode(
                add_eos([item[0] for item in query_answer_span_list]),
                show_progress_bar=True,
                batch_size=batch_size * 3,
                normalize_embeddings=True,
                prompt=query_prefix,
            )
            p_vecs = model.encode(
                add_eos(passage_list),
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "gte-Qwen2" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                show_progress_bar=True,
                batch_size=batch_size * 3,
                normalize_embeddings=True,
                prompt_name="query",
            )
            p_vecs = model.encode(
                passage_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "infgrad/very_awesome" in model_name:
            model.max_seq_length = 32 * 1024
            RETRIEVE_Q_PROMPT = "<|START_INSTRUCTION|>Answer the question<|END_INSTRUCTION|>"
            RETRIEVE_P_PROMPT = "<|START_INSTRUCTION|>Candidate document<|END_INSTRUCTION|>"

            q_vecs = model.encode(
                [f"{RETRIEVE_Q_PROMPT}{q}" for q in [item[0] for item in query_answer_span_list]],
                show_progress_bar=True,
                batch_size=batch_size * 3,
                normalize_embeddings=True
            )
            p_vecs = model.encode(
                [f"{RETRIEVE_P_PROMPT}{p}" for p in passage_list],
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True
            )
        elif "api" in model_name:
            model_name = model_name.split("_")[1]
            if model_name == "openai":
                encoder = commercial_encoder_api.OpenAIEncoder()
                q_vecs = encoder.encode(
                    sentences=[item[0] for item in query_answer_span_list],
                    normalize_embeddings=True,
                )
                p_vecs = encoder.encode(
                    sentences=passage_list,
                    normalize_embeddings=True,
                )
            elif model_name == "cohere":
                encoder = commercial_encoder_api.CohereEncoder()
                q_vecs = encoder.encode(
                    sentences=[item[0] for item in query_answer_span_list],
                    normalize_embeddings=True,
                    prompt_name="search_query",
                )
                p_vecs = encoder.encode(
                    sentences=passage_list,
                    normalize_embeddings=True,
                    prompt_name="search_document",
                )
            elif model_name == "voyage":
                encoder = commercial_encoder_api.VoyageEncoder()
                q_vecs = encoder.encode(
                    sentences=[item[0] for item in query_answer_span_list],
                    normalize_embeddings=True,
                    prompt_name="query",
                    output_dimension=2048,
                )
                p_vecs = encoder.encode(
                    sentences=passage_list,
                    normalize_embeddings=True,
                    prompt_name="document",
                    output_dimension=2048,
                )

            elif model_name == "jina":
                encoder = commercial_encoder_api.JinaEncoder()
                q_vecs = encoder.encode(
                    sentences=[item[0] for item in query_answer_span_list],
                    normalize_embeddings=True,
                    prompt_name="retrieval.query",
                    output_dimension=1024,
                )
                p_vecs = encoder.encode(
                    sentences=passage_list,
                    normalize_embeddings=True,
                    prompt_name="retrieval.passage",
                    output_dimension=1024,
                )
        else:
            raise Exception(f"unsupported model {model_name}")

        # search topk
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
            
        if model:
            del model
