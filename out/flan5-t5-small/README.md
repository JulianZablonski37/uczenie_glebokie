---
language:
- text
- label
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- bleu
- accuracy
model-index:
- name: flan-t5-small
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-small

This model is a fine-tuned version of [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3336
- Bleu: 0.0
- Accuracy: 0.732
- Gen Len: 2.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- training_steps: 2500

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu | Accuracy | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:----:|:--------:|:-------:|
| 0.4112        | 0.22  | 250  | 0.3999          | 0.0  | 0.6503   | 2.0033  |
| 0.3495        | 0.44  | 500  | 0.3746          | 0.0  | 0.6928   | 2.0033  |
| 0.3733        | 0.67  | 750  | 0.3510          | 0.0  | 0.6928   | 2.0     |
| 0.3728        | 0.89  | 1000 | 0.3420          | 0.0  | 0.7157   | 2.0     |
| 0.3388        | 1.11  | 1250 | 0.3470          | 0.0  | 0.7059   | 2.0     |
| 0.3257        | 1.33  | 1500 | 0.3476          | 0.0  | 0.6928   | 2.0     |
| 0.3343        | 1.56  | 1750 | 0.3572          | 0.0  | 0.6993   | 2.0     |
| 0.3364        | 1.78  | 2000 | 0.3336          | 0.0  | 0.732    | 2.0     |
| 0.3319        | 2.0   | 2250 | 0.3511          | 0.0  | 0.7059   | 2.0     |
| 0.2985        | 2.22  | 2500 | 0.3461          | 0.0  | 0.7026   | 2.0     |


### Framework versions

- Transformers 4.23.1
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2
