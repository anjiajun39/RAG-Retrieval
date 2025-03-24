# Installation Environment

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid incompatibility between the automatically installed torch and the local cuda, it is recommended to manually install a torch compatible with the local cuda version before proceeding to the next step.
pip install -r requirements.txt 
```

| Requirement | Recommend |
| ---------------| ---------------- |
| accelerate    |             1.0.1 |
| deepspeed | 0.15.4|
| transformers | 4.44.2|          

# Fine-tuning the Model

After installing the dependencies, we will demonstrate through specific examples how to use our own data to fine-tune the open-source ranking model (BAAI/bge-reranker-v2-m3), or train a ranking model from scratch using BERT-like models (hfl/chinese-roberta-wwm-ext) and LLM-like models (Qwen/Qwen2.5-1.5B). At the same time, we also support distilling the ranking ability of LLM-like models into smaller BERT models.

## Data Format

We support the following standard data format:
```
{"query": str, "hits": [{"content": xxx, "label_1": xxx, "label_2": xxx}, ...]}
```
- `hits` represents all document samples under the query, and `content` is the actual content of the document.
- `label_1/2/...` represents the relevance labels assigned through manual annotation or scoring by a teacher model, serving as the supervision signal for model fine-tuning.

## Data Loading

We provide two data loading methods to support different types of loss functions:

**Single-point Data Loading** supports Mean Squared Error (`MSE`) and Binary Cross Entropy loss, aiming to optimize the absolute relevance judgment of a single query-content point. You need to manually specify the relevance labels used in the dataset: `train_label_key` and `val_label_key`. When the relevance is a multi-level label, by setting `max_label` and `min_label`, the dataset will automatically scale the multi-level labels uniformly to the 0-1 score interval. For example, if there are three-level labels (0, 1, 2) in the dataset, after scaling, we get { label 0: 0, label 1: 0.5, label 2: 1}. During prediction, the final prediction score of the model is the logit output by the model, which can be normalized to the 0-1 interval using the sigmoid function later. Users can use an LLM to obtain relevance labels for distillation. You can find the code for using an LLM to score and annotate relevance in the [examples/distill_llm_to_bert](../../../examples/distill_llm_to_bert) directory.

``` 
train_dataset: "../../../example_data/pointwise_reranker_train_data.jsonl"
max_label: 2
min_label: 0
max_len: 512
shuffle_rate: 0.0
train_label_key: "label"
val_dataset: "../../../example_data/pointwise_reranker_eval_data.jsonl"
val_label_key: "label"
```

**Grouped Data Loading** supports Pairwise RankNet Loss and Listwise Cross Entropy loss, aiming to optimize the relative relevance judgment of query-list[content]. `train_group_size` sets how many documents' relative relevance need to be considered simultaneously for each query, and if the number of original documents is insufficient, repeated sampling will be performed. `train_label_key` and `val_label_key` specify the source of the supervised signal, which can be manual annotation or listwise ranking using an advanced language model.

```
train_dataset: "../../../example_data/grouped_reranker_train_data.jsonl"
train_dataset_type: "grouped"
train_label_key: "gpt4o_listwise"
train_group_size: 10
shuffle_rate: 0.0
max_len: 128
val_dataset: ../../../example_data/grouped_reranker_eval_data.jsonl"
val_dataset_type: "grouped"
val_label_key: "gpt4o_listwise"
```

## Training

### Training of BERT-like models, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
>./logs/training_bert.log &
```

```

### Training of LLM-based model, deepspeed(zero1-2, not for zero3)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
>./logs/training_llm_deepspeed1.log &
```

### Parameter Explanation

Configuration file for multi-GPU training:

- For BERT-like models, fsdp is used by default to support multi-GPU training. Here is an example of the configuration file.
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml): If you want to train a ranking model from scratch based on hfl/chinese-roberta-wwm-ext, use this configuration file.
  - [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml): If you want to fine-tune on the basis of BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1, BAAI/bge-reranker-v2-m3, use this configuration file, as they are all trained based on the multilingual XLMRoberta.

- For LLM-like models, it is recommended to use deepspeed to support multi-GPU training. Currently, only zero1 and zero2 are supported during the training phase. Here are examples of the configuration files.
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

- Modification of the multi-GPU training configuration file:
  - Modify the CUDA_VISIBLE_DEVICES="0" in the command to the multi-GPUs you want to set.
  - Modify the `num_processes` in the above-mentioned configuration file to the number of GPUs you want to run.

Model-related:
- `model_name_or_path`: The name of the open-source reranker model or the location on the local server where it is downloaded. For example: BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1. You can also train from scratch, such as BERT: hfl/chinese-roberta-wwm-ext and LLM: Qwen/Qwen2.5-1.5B.
- `model_type`: Currently, bert_encoder or llm_decoder type models are supported.
- `max_len`: The maximum input length supported by the data.

Dataset-related:
- `train_dataset`: The training dataset, with the format described above.
- `val_dataset`: The validation dataset, with the same format as the training dataset (if there is none, set it to None).
- `max_label`: The maximum label in the dataset, with a default value of 1.
- `min_label`: The minimum label in the dataset, with a default value of 0.

Training-related:
- `output_dir`: The directory for saving the checkpoints during training and the final model.
- `loss_type`: Choose from point_ce (Cross Entropy Loss) and point_mse (Mean Squared Error Loss).
- `epoch`: The number of epochs the model is trained on the training dataset.
- `lr`: The learning rate, generally between 1e-5 and 5e-5.
- `batch_size`: The number of query-doc pairs in each batch.
- `seed`: Set a unified seed for reproducibility of experimental results.
- `warmup_proportion`: The proportion of the number of learning rate warm-up steps to the total number of model update steps. If set to 0, no learning rate warm-up will be performed, and the cosine annealing will start directly from the set `lr`.
- `stable_proportion`: The proportion of the number of steps where the learning rate remains stable to the total number of model update steps, with a default value of 0.
- `gradient_accumulation_steps`: The number of gradient accumulation steps. The actual batch_size of the model is equal to `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`.
- `mixed_precision`: Whether to perform mixed-precision training to reduce the memory requirement. Mixed-precision training optimizes memory usage by using low precision for calculations and high precision for parameter updates. And bf16 (Brain Floating Point 16) can effectively reduce abnormal situations of loss scaling, but this type is only supported by some hardware.
- `save_on_epoch_end`: Whether to save the model at the end of each epoch.
- `num_max_checkpoints`: Control the maximum number of checkpoints saved in a single training session.
- `log_interval`: The model records the loss every x parameter updates.
- `log_with`: Visualization tools, choose from wandb and tensorboard.

Model parameters:
- `num_labels`: The number of logits output by the model, that is, the number of classification categories of the model.
- When an LLM is used for discriminative ranking scoring, the input format needs to be manually constructed, introducing the following parameters:
  - `query_format`, e.g. "query: {}"
  - `document_format`, e.g. "document: {}" 
  - `seq`: Separate the query and document parts, e.g. " "
  - `special_token`: Indicate the end of the document content and guide the model to start scoring. Theoretically, it can be any token, e.g. "\</s>" 
  - The overall format is: "query: xxx document: xxx\</s>" 

# Loading the Model for Prediction

For the saved model, you can easily load the model for prediction.

Cross-Encoder model (BERT-like)
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"]
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

LLM-Decoder model (Scalar mapping based on MLP)

> To meet the special case of using an LLM such as "Qwen/Qwen2.5-1.5B" for discriminative ranking, a relevant format has been designed. The actual effect is: "query: {xxx} document: {xxx}\</s>". Experiments have shown that the introduction of \</s> significantly improves the ranking performance of the LLM [from https://arxiv.org/abs/2411.04539 section 4.3].

```python
ckpt_path = "./Qwen2-1.5B-Instruct"
reranker = LLMDecoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
    query_format="query: {}",
    document_format="document: {}",
    seq="\n",
    special_token="\nrelevance",
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```