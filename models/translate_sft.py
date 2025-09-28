from unsloth import FastLanguageModel, is_bf16_supported, add_new_tokens
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
from datasets import load_dataset
import wandb
import argparse
import torch
import os

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_lang', type=str, help='Prompt language', default='en')
    return parser

parser = arg_parser()
parse = parser.parse_args()

prompt_lang = parse.prompt_lang

login(os.environ['HUGGINGFACE_TOKEN'])

wandb.init(
    project='Translation-SFT',
    name='Mistral-CPT-0.2-en-ms-SFT'
)

max_seq_length = 4096
dtype=None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/mistral-7b-v0.3-bnb-4bit',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

tokenizer = 

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules='all-linear',
    lora_alpha=4,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    use_rslora=True,
    loftq_config=None
)

en_prompt = """<BOI>Translate the following text to the Malay language.
{text}<EOI>

<BOT>
{teks}
<EOT>
"""

ms_prompt = """<BOI>Terjemahkan teks berikut kepada Bahasa Melayu.
{text}<EOI>

<BOT>
{teks}
<EOT>
"""