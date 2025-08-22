from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import json

# Load dataset
train_data = load_dataset("json", data_files="data/train_2.jsonl")["train"]
eval_data = load_dataset("json", data_files="data/eval_2.jsonl")["train"]

# Load tokenizer and model
model_name = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess: tokenize input/output
def preprocess(example):
    input_text = f'Extract fields: {example["input"]}'
    output_text = json.dumps(example["output"], ensure_ascii=False)
    return tokenizer(input_text, text_target=output_text, max_length=512, truncation=True)

train_data = train_data.map(preprocess, remove_columns=train_data.column_names)
eval_data = eval_data.map(preprocess, remove_columns=eval_data.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training args
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='logs',
    learning_rate=3e-4,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model + tokenizer
trainer.save_model("t5_autofill_model")
tokenizer.save_pretrained("t5_autofill_model")
