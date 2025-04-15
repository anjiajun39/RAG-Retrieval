# ( python exp3_1K_evaluate_recall_with_different_span.py >> exp3_1k.log &)


# 定义模型列表
all_models=(
    # "jinaai/jina-embeddings-v3"
    # "BAAI/bge-m3"
    "NovaSearch/jasper_en_vision_language_v1"
    "nvidia/NV-Embed-v2"
    "Alibaba-NLP/gte-Qwen2-7B-instruct"
    "api_voyage"
    "api_openai"
)

# 循环遍历模型列表
for model in "${all_models[@]}"; do
    echo "Evaluating model: $model"
    python exp3_8K_evaluate_recall_with_different_span.py "$model" >> $CHECKPOINT_SAVE/exp3_8K.log
done