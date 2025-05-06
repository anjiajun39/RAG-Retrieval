import numpy as np
from typing import Literal
from sentence_transformers import SentenceTransformer
import commercial_embedding_api
import faiss
import torch

def find_topk_via_faiss(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)
    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance

def find_topk_by_single_vecs(
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find topk passages for each query using single vector.
    Args:
        embedding_model_name_or_path: model name or path
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    if model_type == "local":
        model = SentenceTransformer(
            embedding_model_name_or_path, trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.float16,
            }
        )
        print(f"Model loaded from {embedding_model_name_or_path}")
        print(f"Model dtype: {next(model.named_parameters())[1].dtype}")
    elif model_type == "api":
        pass
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if "jina-embeddings-v3" in embedding_model_name_or_path:
        model.max_seq_length = 8192     
        q_vecs = model.encode(
            query_list,
            task="retrieval.query",
            prompt_name="retrieval.query",
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
        )
        p_vecs = model.encode(
            passage_list,
            task="retrieval.passage",
            prompt_name="retrieval.passage",
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
    elif "bge-m3" in embedding_model_name_or_path or "jina-embeddings-v2-base-en" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "jasper_en_vision_language_v1" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            prompt_name="s2p_query",
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=8,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "stella_en_400M_v5" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            prompt_name="s2p_query",
            batch_size=32,
            chunk_size=512,
            normalize_embeddings=True,
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=8,
            chunk_size=512,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "nvidia" in embedding_model_name_or_path:
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

        model.max_seq_length = 8192
        # model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            add_eos(query_list),
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
            prompt=query_prefix,
        )
        p_vecs = model.encode_multi_process(
            add_eos(passage_list),
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif "gte-Qwen2" in embedding_model_name_or_path:
        model.max_seq_length = 8192
        pool = model.start_multi_process_pool() 
        q_vecs = model.encode_multi_process(
            query_list,
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
            prompt_name="query",
        )
        p_vecs = model.encode_multi_process(
            passage_list,
            pool,
            show_progress_bar=True,
            batch_size=2,
            chunk_size=256,
            normalize_embeddings=True,
        )
        model.stop_multi_process_pool(pool)
    elif model_type == "api":
        if "openai" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.OpenAIEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
            )
        elif "cohere" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.CohereEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
                normalize_embeddings=True,
                prompt_name="search_query",
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="search_document",
            )
        elif "voyage" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.VoyageEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
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

        elif "jina" in embedding_model_name_or_path:
            encoder = commercial_embedding_api.JinaEncoder()
            q_vecs = encoder.encode(
                sentences=query_list,
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
            raise Exception(f"Unsupported api model: {embedding_model_name_or_path}")
    else:
        raise Exception(f"Unsupported model: {embedding_model_name_or_path}")
    
    # search topk
    topk_index, topk_scores = find_topk_via_faiss(q_vecs, p_vecs, topk)
    
    return topk_index, topk_scores

def find_topk_by_multi_vecs(
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
) -> np.ndarray:
    """
    Find topk passages for each query using multiple vectors.
    Args:
        embedding_model_name_or_path: model name or path
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    pass

def find_topk_by_reranker(
    reranker_model_name_or_path: str,
    embedding_model_name_or_path: str,
    model_type: Literal["local", "api"],
    query_list: list[str],
    passage_list: list[str],
    topk: int,
    cache_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find topk passages for each query using reranker.
    Args:
        reranker_model_name_or_path: model name or path
        embedding_model_name_or_path: model name or path for first stage retrieval
        model_type: model type, either 'local' or 'api'
        query_list: list of query strings
        passage_list: list of passage strings
        topk: number of top passages to return
        cache_path: path to cache the first stage retrieval results and reranker scores
    Returns:
        topk_index: numpy array of shape (len(query_list), topk)
        topk_scores: numpy array of shape (len(query_list), topk)
    """
    pass
