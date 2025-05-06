export HF_ENDPOINT=https://hf-mirror.com

# python exp3_metric_of_different_span.py \
#     --data_name_or_path exp3_1k_qp_all.jsonl \
#     --model_name_or_path /data/zzy/models/jinaai/jina-embeddings-v3 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp3.log

python exp3_metric_of_different_span.py \
    --data_name_or_path exp3_1k_qp_all.jsonl \
    --model_name_or_path /data/zzy/models/NovaSearch/stella_en_400M_v5 \
    --model_type local \
    --score_type single_vec \
    2>&1 >> run_exp3.log

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