export HF_ENDPOINT="https://hf-mirror.com"
export VOYAGE_KEY=""
export CUDA_VISIBLE_DEVICES=0

# echo "run exp3 bm25"

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --score_type bm25 \
#     2>&1 >> run_exp3.log

# echo "run exp3 single embedding"

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/jinaai/jina-embeddings-v3 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/NovaSearch/stella_en_400M_v5 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/BAAI/bge-m3 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log 

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path voyage \
#     --model_type api \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path openai \
#     --model_type api \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

# echo "run exp3 multi embedding"

python exp3_metric_of_different_span.py \
    --data_name_or_path exp3_1k_qp_all.jsonl \
    --model_name_or_path /data/zzy/models/BAAI/bge-m3 \
    --model_type local \
    --score_type multi_vec \
    2>&1 >> run_exp3.log 

    # --query_sampling \

# echo "run exp3 reranker"

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/BAAI/bge-reranker-v2-m3 \
#     --model_type local \
#     --first_stage_model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --first_stage_model_type local \
#     --score_type reranker \
#     --query_sampling \
#     2>&1 >> run_exp3.log 

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/Alibaba-NLP/gte-multilingual-reranker-base \
#     --model_type local \
#     --first_stage_model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --first_stage_model_type local \
#     --score_type reranker \
#     --query_sampling \
#     2>&1 >> run_exp3.log 

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/jinaai/jina-reranker-v2-base-multilingual \
#     --model_type local \
#     --first_stage_model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --first_stage_model_type local \
#     --score_type reranker \
#     --query_sampling \
#     2>&1 >> run_exp3.log

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path rerank-2 \
#     --model_type api \
#     --first_stage_model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --first_stage_model_type local \
#     --score_type reranker \
#     --query_sampling \
#     2>&1 >> run_exp3.log 

# python exp3_metric_of_different_span.py\
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/BAAI/bge-reranker-v2-gemma \
#     --model_type local \
#     --first_stage_model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --first_stage_model_type local \
#     --score_type reranker \
#     --query_sampling \
#     2>&1 >> run_exp3.log