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
