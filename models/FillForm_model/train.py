import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

# -----------------------------
# MPS (Apple Silicon) safe setup
# -----------------------------
USE_MPS = torch.backends.mps.is_available()
if USE_MPS:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # ปรับ watermark ให้อยู่ในช่วงถูกต้องและเปิด headroom มากขึ้น
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]  = "0.55"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.98"  # ถ้ายัง OOM ค่อยตั้ง 1.0 หรือ 0.0 (เสี่ยง)
device = torch.device("mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[Info] Using device: {device}")

# -----------------------------
# Load dataset
# -----------------------------
train_data = load_dataset("json", data_files="data/train_3.jsonl", split="train")
eval_data  = load_dataset("json", data_files="data/eval_3.jsonl",  split="train")

# -----------------------------
# Load tokenizer and model
# -----------------------------
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name, use_fast=False)

# โหลดน้ำหนักเป็น FP16 (ไม่ใช่ mixed precision ของ Accelerate)
model = MT5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if USE_MPS else None,
    low_cpu_mem_usage=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.to(device)

# -----------------------------
# Serialize & preprocess
# -----------------------------
MAX_IN_LEN = 192   # ลดลงเพื่อกัน OOM
MAX_TGT_LEN = 16

def serialize_record(record):
    f1 = record.get("form_1", {}) or {}
    f2 = record.get("form_2", {}) or {}
    car = f1.get("car_info", {}) or {}
    driver = f1.get("driver_info") or f2.get("driver_info") or {}
    location = f2.get("location", {}) or {}

    return (
        f"Case {f1.get('case_id', '')}, Accident {f2.get('accident_id', '')}. "
        f"Driver {driver.get('name', '')} {driver.get('surname', '')}, "
        f"age {driver.get('age', '')}, injury {driver.get('injury', '')}. "
        f"Car {car.get('manufacturer', '')} {car.get('model', '')}, "
        f"year {car.get('model_year', '')}, type {car.get('type', '')}. "
        f"Location: {location.get('province', '')}, {location.get('district', '')}, {location.get('subdistrict', '')}."
    )

def preprocess_function(examples):
    inputs, targets = [], []
    for f1, f2 in zip(examples["form_1"], examples["form_2"]):
        inputs.append(serialize_record({"form_1": f1, "form_2": f2}))
        targets.append(f2.get("accident_id", ""))

    # ใช้ dynamic padding ภายใน batch + จัดให้เป็น multiple of 8 ช่วย allocator
    model_inputs = tokenizer(inputs, max_length=MAX_IN_LEN, truncation=True)
    labels = tokenizer(targets, max_length=MAX_TGT_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_eval  = eval_data.map(preprocess_function,  batched=True, remove_columns=eval_data.column_names)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    pad_to_multiple_of=8,   # ลด fragmentation ใช้เมมได้เสถียรกว่า
)

# -----------------------------
# Callback: เคลียร์ cache ทุกสเต็ป
# -----------------------------
class MPSEmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if USE_MPS:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

# -----------------------------
# TrainingArguments — ประหยัดแรม
# -----------------------------
training_args = TrainingArguments(
    output_dir="./outputs",
    eval_strategy="no",        # ปิด eval ระหว่างเทรน เพื่อลด peak mem (ค่อย eval ทีหลัง)
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=6,
    gradient_accumulation_steps=32,  # เพิ่มเพื่อ effective batch โดยไม่เพิ่ม VRAM
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    optim="adafactor",               # ใช้ Adafactor ประหยัดแรมกว่า AdamW มาก
    fp16=False,                      # ห้ามเปิด (Accelerate จะ error บน MPS)
    bf16=False,
    torch_compile=False,
    group_by_length=True,            # รวม sequence ใกล้เคียง ลด padding => ลดเมม
    report_to="none",
    seed=42,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,   # แม้ปิด eval ระหว่างเทรน ยังส่งไว้เพื่อใช้ตอน trainer.evaluate() ภายหลัง
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[MPSEmptyCacheCallback()],
)

if __name__ == "__main__":
    # เคลียร์ก่อนเริ่ม
    if USE_MPS:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    trainer.train()

    out_dir = "t5_autofill_model"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Info] Saved model + tokenizer to: {out_dir}")

    # (ถ้าต้องการประเมินหลังจบ เพื่อไม่ให้ชนกับ mem ระหว่างเทรน)
    # results = trainer.evaluate()
    # print(results)
