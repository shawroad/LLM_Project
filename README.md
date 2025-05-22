# LLM_Project
This repository will include technologies related to large models, such as fine-tuning of large language models, the use of agents, and other large model-related techniques.
```
├── DPO_Train_From_Scratch   # DPO实现
│   ├── config.py
│   ├── data
│   │   └── readme.txt
│   ├── data_helper.py
│   └── run_train.py
├── DPO_Train_Use_Trainer   # 使用huggingface实现的DPOTrainer
│   ├── data
│   │   └── readme.txt
│   └── run_train.py
├── GRPO_Train_From_Scratch
│   └── run_train.py
├── GRPO_Train_Use_Trainer
│   ├── Qwen2.5-0.5B-Instruct-GRPO
│   └── run_train.py
├── PPO_Train_From_Scratch
│   ├── ppo_model_output
│   │   ├── config.json
│   │   └── generation_config.json
│   └── run_train.py
├── README.md
├── SFT_Train_From_Scratch
│   ├── config.py
│   ├── data
│   │   ├── test_distill_r1_110k_sft.json
│   │   └── train_distill_r1_110k_sft.json
│   ├── data_helper.py
│   ├── inference.py
│   ├── run_train.py
│   └── sft_model_output
│       ├── events.out.tfevents.1745750089.model.2109.0
│       └── logs.txt
├── SFT_Train_Use_Trainer
│   ├── Dynamic_Padding
│   │   ├── inference.py
│   │   ├── run_train.py
│   │   └── sft_model_output
│   │       └── events.out.tfevents.1745910597.model.20787.1
│   └── No_Dynamic_Padding
│       ├── inference.py
│       ├── run_train.py
│       └── sft_model_output
│           └── events.out.tfevents.1745900707.model.33343.1
└── SFT_Train_Use_Unsloth
    └── run_train.py
```
