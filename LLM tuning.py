##Setting up Hugging Face
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
!pip install huggingface_hub

import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging)

##Import the datesets
model_identfier = "aboonaji/wiki_medical_terms_llam2_format"
source_dataset = "gamino/wiki_medical_terms"
formatted_dataset = "aboonaji/wiki_medical_terms_llam2_format"

##QlOra HyperParameters
lora_hyper_r = 64
lora_hyper_alpha = 16
lora_hyper_dropout  = 0.1

##BitsandBytes Parameters
enable_4Bit = True
compute_dtype_bnb = "float16"
quant_type_bnb = "nf4"
double_quant_flag = False

###training arguments_hyperparameters
results_dir = "./results"
epochs_count = 10
enable_fp16 = False
enable_bf16 = False
train_batch_size = 4
eval_batch_size = 4
accumulation_steps = 1
checkpointing_flag = True
grad_norm_limit = 0.3
train~_learning_rate = 2e-4
decay_rate = 0.001
optimizer_type = "paged_adamw_32bit"
scheduler_type ="cosine"
steps_limit = 100
warmup_percentage = 0.03

##fine tuning Hyperparameters
enable_packing = False
sequence_lenght_max = None
device_assignment = {"": 0}

###loading the dataset
training_data = load_dataset(formatted_dataset, split = "train")
training_data
Dataset({
  features ['text],
  num_rows: 6861
})





