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
    target_len=7000,
    mode="before",
    tokenizer=None,
):
    """
    拼接文本，使总长度接近 target_len（按单词数），支持三种拼接方式。
    本版本不再分句，而是直接使用整个文本块，同时确保同一文本块不会重复使用。

    Args:
        content (str): 原始文本内容。
        preprocessed_texts (List[Tuple[str, int]]): 已处理好的 (text, word_count) 对。
        target_len (int): 目标总词数。
        mode (str): content 所在位置，可选值为 "before", "middle", "after"。
        tokenizer: 分词器，如果提供则使用分词器计算长度，否则按空格分割计算长度。

    Returns:
        str: 拼接好的文本字符串。
    """
    # 验证 mode 参数
    if mode not in ["before", "middle", "after"]:
        raise ValueError("mode 参数必须是 'before', 'middle' 或 'after'")

    if tokenizer:
        content_len = len(tokenizer.tokenize(content))
    else:
        content_len = len(content.split())
    remaining_len = target_len - content_len
    if remaining_len <= 0:
        return content

    def collect_samples(target_words):
        total = 0
        samples = []
        # 避免 ValueError 异常
        sample_size = min(len(preprocessed_texts), 100)
        for text, count in random.sample(preprocessed_texts, sample_size):
            # 如果整个文本块加入后不会超过目标词数，则直接添加
            if total + count <= target_words:
                samples.append(text)
                total += count
            else:
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
    from transformers import AutoTokenizer

    data_dir = "/processing_data/search/zengziyang/data/"
    model_dir = "/processing_data/search/zengziyang/models/"
    
    # model_name = "nvidia/NV-Embed-v2"
    model_name = sys.argv[1]
    model_name = model_dir + model_name

    base_dir = ""
    batch_size = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir + "nvidia/NV-Embed-v2")
    
    text_set = set()
    parquet_data = pl.read_parquet(data_dir + "fineweb-edu/000_00000.parquet").rows(named=True)
    for document in tqdm.tqdm(parquet_data):
        if 512 <= len(document["text"].split()) <= 2048:
            text_set.add(document["text"])
    text_set = list(text_set)
    import multiprocessing

    with multiprocessing.Pool() as pool:
        text_lens = [len(ids["input_ids"]) for ids in pool.map(tokenizer, text_set)]
    
    text_lens = {p: l for p, l in zip(text_set, text_lens)}
    text_set = set([p for p in text_set if 500 <= text_lens[p] <= 2000])

    print("number of all texts from fineweb-edu/000_00000.parquet", len(text_set))
    
    preprocessed_texts = [(p, text_lens[p]) for p in text_set]

    
    train_data_path = "exp3_qp_train.jsonl"
    test_data_path = "exp3_qp_test.jsonl"

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
                model_name, trust_remote_code=True, cache_folder=None
            )
            pool = model.start_multi_process_pool()
    else:
        import commercial_encoder_api
        model = None
        pool = None
    
    for mode in ["before", "middle", "after"]:
        passage_list_file = f"exp3_8K_passages_{mode}.json"
        if os.path.exists(passage_list_file):
            print(f"loading {passage_list_file} ...")
            with open(passage_list_file) as f:
                passage_list = json.load(f)
        else:
            print(f"generating {passage_list_file} ...")
            passage_list = [pad_content_with_samples(p, preprocessed_texts, mode=mode, target_len=7000, tokenizer=tokenizer) for p in raw_passage_list]
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