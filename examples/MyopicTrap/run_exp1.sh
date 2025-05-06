export HF_ENDPOINT=https://hf-mirror.com


# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path /data/zzy/models/jinaai/jina-embeddings-v3 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log

python exp1_metric_of_different_answer_position.py \
    --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
    --model_name_or_path /data/zzy/models/NovaSearch/stella_en_400M_v5\
    --model_type local \
    --score_type single_vec \
    2>&1 >> run_exp1.log

# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path /data/zzy/models/nvidia/NV-embed-v2 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log

# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path /data/zzy/models/Alibaba-NLP/gte-Qwen2-7B-instruct \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log

# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path /data/zzy/models/BAAI/bge-m3 \
#     --model_type local \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log 

# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path voyage \
#     --model_type api \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log

# python exp1_metric_of_different_answer_position.py \
#     --data_name_or_path /data/zzy/data/rajpurkar/squad_v2\
#     --model_name_or_path openai \
#     --model_type api \
#     --score_type single_vec \
#     2>&1 >> run_exp1.log