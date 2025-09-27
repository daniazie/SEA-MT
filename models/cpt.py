from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from transformers import TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
import torch
import os

login(os.environ['HUGGINGFACE_TOKEN'])

max_seq_length = 4096
dtype=None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Mistral-Small-3.1-24B-Base-2503-bnb-4bit',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    use_rslora=True,
    loftq_config=True
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
dataset = dataset.map(formatting_prompts_func, batched=True)

os.makedirs('model_exp', exist_ok=True)
os.makedirs('model_exp/Mistral-ms-CPT', exist_ok=True)

args = UnslothTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    warmup_ratio=0.1,
    num_train_epochs=10,
    learning_rate=5e-5,
    embedding_learning_rate=1e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
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

model.push_to_hub("daniazie/Mistral-3.1-24B-ms-CPT-adapters", tokenizer, save_method = "merged_4bit")
model.push_to_hub_merged("daniazie/Mistral-3.1-24B-ms-CPT", tokenizer, save_method = "merged_4bit")