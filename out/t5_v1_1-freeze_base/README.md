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
- name: t5_v1_1_freeze_base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5_v1_1_freeze_base

This model is a fine-tuned version of [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3256
- Bleu: 0.0
- Accuracy: 0.7092
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
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu | Accuracy | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:----:|:--------:|:-------:|
| 11.4757       | 0.22  | 250  | 0.6369          | 0.0  | 0.4542   | 2.0     |
| 0.8998        | 0.44  | 500  | 0.5513          | 0.0  | 0.451    | 2.0     |
| 0.7388        | 0.67  | 750  | 0.5040          | 0.0  | 0.451    | 2.0     |
| 0.6701        | 0.89  | 1000 | 0.4202          | 0.0  | 0.6176   | 2.0     |
| 0.5685        | 1.11  | 1250 | 0.5437          | 0.0  | 0.6176   | 2.0     |
| 0.4571        | 1.33  | 1500 | 0.5718          | 0.0  | 0.6373   | 2.0     |
| 0.4495        | 1.56  | 1750 | 0.4097          | 0.0  | 0.6863   | 2.0     |
| 0.414         | 1.78  | 2000 | 0.3948          | 0.0  | 0.6699   | 2.0     |
| 0.4066        | 2.0   | 2250 | 0.4504          | 0.0  | 0.6699   | 2.0     |
| 0.3926        | 2.22  | 2500 | 0.3600          | 0.0  | 0.6928   | 2.0     |
| 0.3913        | 2.44  | 2750 | 0.3782          | 0.0  | 0.6895   | 2.0     |
| 0.3558        | 2.67  | 3000 | 0.3492          | 0.0  | 0.6895   | 2.0     |
| 0.3955        | 2.89  | 3250 | 0.3657          | 0.0  | 0.6732   | 2.0     |
| 0.3496        | 3.11  | 3500 | 0.3617          | 0.0  | 0.6961   | 2.0     |
| 0.3452        | 3.33  | 3750 | 0.3704          | 0.0  | 0.7059   | 2.0     |
| 0.3647        | 3.56  | 4000 | 0.3521          | 0.0  | 0.7059   | 2.0     |
| 0.3322        | 3.78  | 4250 | 0.3256          | 0.0  | 0.7092   | 2.0     |
| 0.3208        | 4.0   | 4500 | 0.3617          | 0.0  | 0.6928   | 2.0     |
| 0.3382        | 4.22  | 4750 | 0.3382          | 0.0  | 0.7026   | 2.0     |
| 0.3355        | 4.44  | 5000 | 0.3297          | 0.0  | 0.7092   | 2.0     |
| 0.3303        | 4.67  | 5250 | 0.3462          | 0.0  | 0.6928   | 2.0     |
| 0.3305        | 4.89  | 5500 | 0.3412          | 0.0  | 0.6928   | 2.0     |


### Framework versions

- Transformers 4.23.1
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2
