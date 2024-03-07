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
train_learning_rate = 2e-4
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

##QLora Config
dtype_computation = getattr(torch, compute_dtype_bnb)
bnb_setup = BitsandBytesConfig(load_in_4Bit = enable_4Bit,
                               bnb_4Bit_quant_type = quant_type_bnb,
                               bnb_4Bit_use_double_quant = double_quant_flag,
                               bnb_4Bit_compute_dtype = dtpye_computation)

##loading the llama models
llama_model = AutoModelForCausalLM.from_pretrained(model_identfier, quantization_config = bnb_setup, device_map = device_assignment)
llama_model.config.use_case = False
llama_model.config.pretraining_tp = 1

##Pretrained Tokenizer
llama_tokenizer = tokenizer.from_pretrained(model_identfier, trust_remote_code = True)
llama_tokenizer.pad_token = llama.tokenizer.eos_token
llama_tokenizer.padding_side = "right"

##Lora Fine tuning
peft_setup = LoraConfig(lora_alpha = lora_hyper_alpha,
                        lora_dropout = lora_hyper_dropout,
                        lora_r = lora_hyper_r,
                        bias = "none",
                        task_type = "CASUAL_LM")

##Training Arguments
train-args =TrainingArguments(output_dir = results_dir,
                              num_train_epochs = epochs_count,
                              per_device_train_batch_size = train_batch_size,
                              per_device_eval_batch_size = teval_batch_size,
                              gradient_accumulation_steps = accumulation_steps,
                              learning_rate = train_learning_rate,
                              weight_decay = decay_rate,
                              optim = optimizer_type,
                              save_steps = checkpoint_interval,
                              logging_steps = log_interval, 
                              fp16 = enable_fp16,
                              bf1 = enable_bf1,
                              max_grad_norm = grad-norm_limit,
                              max_steps = steps_limit,
                              warmup_ratio = warmup_percentage,
                              group_by_lenght = lenght_grouping,
                              lr_scheduler_type = scheduler_type,
                              gradient_checkpointing = checkpointing_flag)


##Supervised Fine Tuning
llama_sftt_trainer = SFTTrainer(model = llama_model,
                                args = train_args,
                                training_dataset = training_data,
                                tokenizer = llama_tokenizer,
                                peft_config = peft_setup,
                                dataset_text_field = "text"
                                max_seq_length = sequence_length_max,
                                packing = enable_packing)

##Training the Model
llama_sftt_trainer.train()
              


