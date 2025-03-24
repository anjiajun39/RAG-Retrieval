import polars as pl
import tqdm
import random
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
import json
import torch

def pad_content_with_samples(
    content,
    preprocessed_texts,
    target_len=8000,
    mode="before",
    allow_truncate=True,
):
    """
    拼接文本，使总长度接近 target_len（按单词数），支持三种拼接方式。
    本版本不再分句，而是直接使用整个文本块，同时确保同一文本块不会重复使用。

    Args:
        content (str): 原始文本内容。
        preprocessed_texts (List[Tuple[str, int]]): 已处理好的 (text, word_count) 对。
        target_len (int): 目标总词数。
        mode (str|None): content 所在位置，可选值为"before","middle", "after"。
        allow_truncate (bool): 是否允许截断最后一个片段以填满目标长度。

    Returns:
        str: 拼接好的文本字符串。
    """
    content_words = content.split()
    content_len = len(content_words)
    remaining_len = target_len - content_len
    if remaining_len <= 0:
        return content


    # 用于去重文本块
    used_texts = set()

    def collect_samples(target_words):
        total = 0
        samples = []
        for text, count in preprocessed_texts:
            if text in used_texts:
                continue

            # 如果整个文本块加入后不会超过目标词数，则直接添加
            if total + count <= target_words:
                samples.append(text)
                total += count
                used_texts.add(text)
            # 如果允许截断且当前总词数不足，则截取部分单词
            elif allow_truncate and total < target_words:
                take = target_words - total
                words = text.split()
                partial = ' '.join(words[:take])
                samples.append(partial)
                total += take
                break

            if total >= target_words:
                break
        return samples

    if mode == "middle":
        left_len = remaining_len // 2
        right_len = remaining_len - left_len
        left_samples = collect_samples(left_len)
        right_samples = collect_samples(right_len)
        final_parts = left_samples + [content] + right_samples
    elif mode == "before":
        final_parts = [content] + collect_samples(remaining_len)
    elif mode == "after":
        final_parts = collect_samples(remaining_len) + [content]

    return "\n".join(final_parts)


def find_topk_by_vecs(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)

    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


if __name__ == "__main__":
    import sys
    
    text_list = []
    parquet_data = pl.read_parquet("fineweb-edu/000_00000.parquet").rows(named=True)
    for document in tqdm.tqdm(parquet_data):
        if 512 <= len(document["text"].split()) <= 2048:
            text_list.append(document["text"])

    print(len(text_list))
    preprocessed_texts = [(t, len(t.split())) for t in text_list]

    data_cache_dir = ""
    model_cache_dir = ""

    base_dir = ""
    batch_size = 5
    
    
    model_name = "nvidia/NV-Embed-v2"
    # model_name = sys.argv[1]
    
    data_base_dir = "examples/short-sighted_pitfalls/"
    train_data_path = data_base_dir + "exp3_qp_train.jsonl"
    test_data_path = data_base_dir + "exp3_qp_test.jsonl"

    topk_list = [1, 5, 10, 20, 30, 50]

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
            
    raw_passage_list = list(set(passage_list))
    raw_passage_list.sort()
    
    passage2id = {passage: idx for idx, passage in enumerate(raw_passage_list)}
    labels = np.array(
        [[passage2id[query2passage[query]]] for query, _ in query_answer_span_list]
    )
    answer_span_list = [answer_span for _, answer_span in query_answer_span_list]
    print("number of all queries", len(query_answer_span_list))
    print("number of all passages", len(raw_passage_list))
    print("min len of passage (words): ", min([len(passage.split(" ")) for passage in raw_passage_list]))
    print("max len of passage (words): ", max([len(passage.split(" ")) for passage in raw_passage_list]))
    
    print(model_name)
    # load model
    if "api" not in model_name:
        if "infgrad/very_awesome" in model_name:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,  # fp16 容易计算出nan
                    "attn_implementation": "flash_attention_2"
                },
                config_kwargs={"single_vector_type": "cls_add_mean"} # mean, cls, cls_add_mean
            ).cuda().bfloat16().eval()
            pool = model.start_multi_process_pool()
        else:
            model = SentenceTransformer(
                model_name, trust_remote_code=True, cache_folder=model_cache_dir
            )
            pool = model.start_multi_process_pool()
    else:
        import commercial_encoder_api
        model = None
        pool = None
    
    for mode in ["before", "middle", "after"]:
        passage_list_file = data_base_dir + f"exp3_8K_passages_{mode}.json"
        if os.path.exists(passage_list_file):
            print(f"loading {passage_list_file} ...")
            with open(passage_list_file) as f:
                passage_list = json.load(f)
        else:
            print(f"generating {passage_list_file} ...")
            passage_list = [pad_content_with_samples(p, preprocessed_texts, mode=mode, target_len=8000) for p in raw_passage_list]
            with open(passage_list_file, "w") as f:
                json.dump(passage_list, f, ensure_ascii=False)

        # get q,p vecs
        if "jina-embeddings-v3" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                task="retrieval.query",
                prompt_name="retrieval.query",
                show_progress_bar=True,
                batch_size=256,
                normalize_embeddings=True,
            )
            p_vecs = model.encode(
                passage_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                batch_size=48,
                normalize_embeddings=True,
            )
        elif "bge-m3" in model_name or "jina-embeddings-v2-base-en" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode_multi_process(
                [item[0] for item in query_answer_span_list],
                pool,
                show_progress_bar=True,
                batch_size=256,
                normalize_embeddings=True,
            )
            p_vecs = model.encode_multi_process(
                passage_list,
                pool,
                show_progress_bar=True,
                batch_size=48,
                normalize_embeddings=True,
            )
            
        elif "jasper_en_vision_language_v1" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode(
                [item[0] for item in query_answer_span_list],
                show_progress_bar=True,
                prompt_name="s2p_query",
                batch_size=128,
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

            # model.max_seq_length = 32768
            model.max_seq_length = 8192
            model.tokenizer.padding_side = "right"
            q_vecs = model.encode_multi_process(
                add_eos([item[0] for item in query_answer_span_list]),
                pool,
                show_progress_bar=True,
                batch_size=8,
                normalize_embeddings=True,
                prompt=query_prefix,
            )
            p_vecs = model.encode_multi_process(
                add_eos(passage_list),
                pool,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            
        elif "gte-Qwen2" in model_name:
            model.max_seq_length = 8192
            q_vecs = model.encode_multi_process(
                [item[0] for item in query_answer_span_list],
                pool,
                show_progress_bar=True,
                batch_size=16,
                normalize_embeddings=True,
                prompt_name="query",
            )
            
            p_vecs = model.encode_multi_process(
                passage_list,
                pool,
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
                batch_size=128,
                normalize_embeddings=True
            )
            p_vecs = model.encode(
                [f"{RETRIEVE_P_PROMPT}{p}" for p in passage_list],
                show_progress_bar=True,
                batch_size=16,
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

        recall_at_k_list = []
        for topk in topk_list:
            recall_at_k_list.append(
                (topk_index[:, :topk] == labels[:, :]).sum()
                / len(labels)
            )
        recall_at_k_list = [str(float(i)) for i in recall_at_k_list]  # for joining
        print(
            f"{model_name}, {len(labels)}, {mode}, {','.join(recall_at_k_list)}"
        )
        
    if model:
        model.stop_multi_process_pool(pool)
        del model
        torch.cuda.empty_cache()
        # UserWarning: resource_tracker: There appear to be 8 leaked semaphore objects to clean up at shutdown