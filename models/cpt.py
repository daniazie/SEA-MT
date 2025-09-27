from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bf16_supported
from transformers import TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
import wandb
import argparse
import torch
import os

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, help='Training set size', default=1.0)
    return parser

parser = arg_parser()
parse = parser.parse_args()

train_size = parse.train_size

login(os.environ['HUGGINGFACE_TOKEN'])
wandb.login(key=os.environ['WANDB_KEY'])

max_seq_length = 4096
dtype=None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/mistral-7b-v0.3-bnb-4bit',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=4,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    use_rslora=True,
    loftq_config=None
)

prompt = """Rencana Wikipedia
### Tajuk: {}


### Kandungan Rencana:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    titles = examples['title']
    texts = examples['text']

    outputs = []

    for title, text in zip(titles, texts):
        text = prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return{'text': outputs}

dataset = load_dataset("wikimedia/wikipedia", '20231101.ms', split='train')
dataset = dataset.train_test_split(train_size=train_size)['train']
dataset = dataset.map(formatting_prompts_func, batched=True)

os.makedirs('model_exp', exist_ok=True)
os.makedirs('model_exp/Mistral-ms-CPT', exist_ok=True)

args = UnslothTrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=512,
    warmup_steps=10,
    warmup_ratio=0.1,
    num_train_epochs=2,
    learning_rate=5e-5,
    embedding_learning_rate=1e-5,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    logging_steps=100,
    optim='adamw_8bit',
    weight_decay=0.01,
    lr_scheduler_type='linear',
    output_dir='model_exp/Mistral-ms-CPT',
    report_to='wandb',
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=args
)

trainer.train()

model.push_to_hub(f"daniazie/Mistral-v0.3-7b-ms-{train_size}-CPT-adapters", tokenizer, save_method = "merged_4bit")
model.push_to_hub_merged(f"daniazie/Mistral-v0.3-7b-ms-{train_size}-CPT", tokenizer, save_method = "merged_4bit")