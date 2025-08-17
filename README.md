# In-Context Transformer Learning + Adaptive Pruning

This repository integrates two codebases to study **what transformers can learn in-context** and **how far pruning alone can adapt a model to a downstream task**:

* **In-Context Learning**: [https://github.com/dtsip/in-context-learning](https://github.com/dtsip/in-context-learning)
  (paper: *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*)
* **Adaptive Pruning (AdaPrune)**: [https://github.com/yg211/bert\_nli](https://github.com/yg211/bert_nli)
  (paper: *Adapting by Pruning: A Case Study on BERT*)

We have refactored the in-context code into a more **PyTorch-idiomatic training pipeline** (configs → dataloaders → trainers → checkpoints) and integrate AdaPrune to **compare fine-tuning vs. pruning-only adaptation** on linear-regression families.

---

## TL;DR

* Pretrain a GPT-2-style transformer on synthetic **linear regression** tasks (with curriculum learning).
* Fine-tune on extended task variants (sparse, dense-zero, tail, noisy).
* Or skip fine-tuning and **adapt by pruning** with AdaPrune (control density via `wanted_density`).
* Stream data loaders (no files): data is generated on-the-fly; buffer-based streaming enables step-wise training.

---

## Repository Structure

```
.
├─ configs/
│  ├─ adaptive_pruning/
│  │  └─ ap_tail_linear_regression.yml        # pruning-only experiments
│  ├─ fine_tuning/
│  │  ├─ ft_noisy_linear_regression.yml
│  │  ├─ ft_scaling_linear_regression.yml
│  │  └─ ft_tail_linear_regression.yml        # fine-tune from a pretrained checkpoint
│  └─ pretraining/
│     ├─ dense_zero_linear_regression.yml
│     ├─ linear_regression.yml                # baseline pretraining
│     └─ sparse_linear_regression.yml
│
├─ resources/                                 # Background material
│  ├─ images/
│  │  ├─ adapting_by_pruning_setting.png
│  │  └─ in_context_learning_setting.jpg
│  └─ paper/
│     ├─ What_Can_Transformers_Learn_In-Context-...pdf
│     └─ Adapting_by_Pruning-A_Case_Study_on_BERT.pdf
│
├─ results/
│  ├─ linear_regression_pretrained/           # example pretrained model
│  └─ logs/                                   # run folders created here (run0, run1, …)
│
├─ src/
│  ├─ data/
│  │  ├─ curriculum.py                        # step-wise curriculum utilities
│  │  ├─ linear_regression.py                 # streaming dataset + loader
│  │  ├─ samplers.py                          # Gaussian + gap samplers, etc.
│  │  ├─ tasks.py                             # Linear regression task variants (sparse/dense-zero/tail/noisy)
│  │  └─ utils.py                             # Some data utilities
│  ├─ model/
│  │  ├─ builder.py                           # constructs GPT-2-like model from YAML
│  │  ├─ mask_model.py                        # AdaPrune ModelMasker (masks + custom opt)
│  │  ├─ nn_model.py, pretrained_model.py     # nn.Module definitions of different transformer and nn models
│  │  └─ transformer_model.py
│  ├─ trainers/
│  │  ├─ backprop_trainer.py                  # step-based pretraining/fine-tuning trainer
│  │  ├─ adaprune_trainer.py                  # step-based pruning-only trainer
│  │  └─ base_trainer.py                      # shared utilities (save/load, evaluate, etc.)
│  └─ utils/
│     ├─ arg_parser.py                        # CLI → args
│     ├─ custom_lr_scheduler.py               # Custom LRS (e.g., step/cosine) for AdaPrune
│     ├─ logger.py                            # simple console/file logger
│     ├─ utils.py                             # seeding, path helpers, etc.
│     └─ worker.py                            # end-to-end orchestration (model/data/trainers)
│
├─ main.py                                    # entrypoint: python main.py --config <cfg>
├─ LICENSE
├─ README.md
└─ requirements.txt                           # Python dependencies
```

---

## Key Ideas

* **PyTorch-native pipeline**
  YAML config → `Worker` builds model/dataloaders/criterion → trainer (backprop or AdaPrune) trains **by steps** (no epochs).

* **Streaming synthetic data**
  `LinearRegressionDataLoader` fills a buffer with freshly generated data from `DynamicDataset`. Buffers refill automatically.

* **Curriculum learning (per step)**
  `Curriculum` updates `n_points` and `n_dims_truncated` at intervals; the loader adapts accordingly (refill on change).

* **Comparing adaptation strategies**

  * **Fine-tuning** (`backprop_trainer`) with standard optimizers & schedulers.
  * **Pruning-only** (`adaprune_trainer`) using **custom optimizer & custom LR scheduler** on mask and weight tensors.

* **Robust checkpointing & resume**
  Resume accepts either a **directory** (picks the latest `stepN.pt`) or a **specific** checkpoint file.

---

## Installation

### Prerequisites

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1) Pretrain on linear regression

```bash
python main.py --config configs/pretraining/linear_regression.yml
```

### 2) Fine-tune on a downstream variant (e.g., tail)

Point the fine-tuning config to your pretrained checkpoint (set `resume_ckpt` or pass `--resume_ckpt`):

```bash
python main.py --config configs/fine_tuning/ft_tail_linear_regression.yml \
               --resume_ckpt results/logs/linear_regression/run0/checkpoints/step500000.pt
```

### 3) Adapt by pruning (no weight fine-tune)

Use AdaPrune trainer and set `wanted_density` (e.g., 0.5 keeps 50% of weights):

```bash
python main.py --config configs/adaptive_pruning/ap_tail_linear_regression.yml \
               --resume_ckpt results/logs/linear_regression/run0/checkpoints/
```

> **Note:** `resume_ckpt` can be a directory (the latest step is chosen) or a specific `.../stepN.pt` file.

---

## Dataloaders

* **No disk datasets**: data is generated with samplers (e.g., Gaussian with optional “gaps”).
* **Buffering**: A buffer of size `batch_size * n_batches` is prefilled; training samples batches sequentially; buffer refills on demand.
* **Validation/Test**: you specify on how many samples to evaluate (`val_steps`, `test_steps`).

---

## Checkpoints & Resuming

* **Layout**: `results/logs/<name>/runX/checkpoints/stepN.pt` + trainer state pkl.
* **Resume** logic reuses the existing run directory when `resume_ckpt` is provided (no new `runX` created).
* **Determinism**: RNG states (Python, NumPy, Torch) are saved/restored to continue where you stopped.

---

## Pretrained Model
You can download a pretrained GPT-2 model for the linear_regression task from the [Releases page](https://github.com/julianscher/gpt-adaprune/releases).

Or download directly with:

```bash
wget https://github.com/julianscher/gpt-adaprune/releases/download/v1.0.0/linear_regression_pretrained.zip
```

Add the downloaded folder to the results folder (```results/linear_regression_pretrained/```) as shown in the repository structure above. 
If the results folder does not exist, create a new one or run your first experiment.

---

## Extending

* Add a new task: implement in `src/data/tasks.py` and register in `get_task_sampler`.
* Add a new sampler (e.g., different distribution or structured gaps): see `src/data/samplers.py`.
* Add a trainer: follow the step-based interface in `src/trainers/backprop_trainer.py` / `adaprune_trainer.py`.

---

## Citing

If you use this repository, please also cite the original works:

* Tsipras et al., *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.*
  Code: [https://github.com/dtsip/in-context-learning](https://github.com/dtsip/in-context-learning)

* Yuval et al., *Adapting by Pruning: A Case Study on BERT.*
  Code: [https://github.com/yg211/bert\_nli](https://github.com/yg211/bert_nli)

---

## License

See [LICENSE](./LICENSE). The integrated components retain their original licenses; please review them when reusing code.

---

## Acknowledgements

Huge thanks to the authors and maintainers of the original repositories for releasing exciting, insightful code and experiments.
