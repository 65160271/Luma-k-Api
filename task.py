import torch
import json
import re
from datetime import datetime, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

app = FastAPI()

# ==== Load Models ====
cls_path_model_dir = "./models/Task/models/classify_model"
cls_path_tokenizer = AutoTokenizer.from_pretrained(cls_path_model_dir)
cls_path_model = AutoModelForSequenceClassification.from_pretrained(cls_path_model_dir)

task_model_dir = "./models/Task/models/classifyTask_model"
task_tokenizer = AutoTokenizer.from_pretrained(task_model_dir)
task_model = AutoModelForSequenceClassification.from_pretrained(task_model_dir)
with open(f"{task_model_dir}/label_names.json", "r", encoding="utf-8") as f:
    task_labels = json.load(f)  # e.g. ["Add","Check","Edit","Remove"]

input_decor_dir = "./models/Task/models/inputDecor_model"
decor_tokenizer = AutoTokenizer.from_pretrained(input_decor_dir)
decor_model = AutoModelForSeq2SeqLM.from_pretrained(input_decor_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decor_model = decor_model.to(device)

# === Thai Time Parsing (quick heuristics)
time_patterns = {
    "ตีห้า": (5, 0), "หกโมงเช้า": (6, 0), "เจ็ดโมง": (7, 0),
    "แปดโมง": (8, 0), "แปดโมงครึ่ง": (8, 30), "เก้าโมง": (9, 0),
    "เก้าโมงครึ่ง": (9, 30), "สิบโมง": (10, 0), "สิบเอ็ดโมง": (11, 0),
    "เที่ยง": (12, 0), "บ่ายโมง": (13, 0), "บ่ายสอง": (14, 0),
    "บ่ายสาม": (15, 0), "สามโมงเย็น": (15, 0), "บ่ายสามครึ่ง": (15, 30),
    "สี่โมงเย็น": (16, 0), "ห้าโมงเย็น": (17, 0), "หกโมงเย็น": (18, 0),
    "หนึ่งทุ่ม": (19, 0),
    "เช้า": (8, 0), "สาย": (10, 0), "สายๆ": (10, 30),
    "เย็น": (17, 30), "ค่ำ": (18, 30)
}

# ====== Schemas ======
class InputText(BaseModel):
    text: str

# ====== Utilities ======
def extract_datetime(text: str):
    now = datetime.now()
    target = now
    found_date, found_time = False, False

    if "พรุ่งนี้" in text:
        target += timedelta(days=1)
        found_date = True

    for phrase, (h, m) in time_patterns.items():
        if phrase in text:
            target = target.replace(hour=h, minute=m)
            found_time = True
            break

    return target if (found_date or found_time) else None

def clean_task_input(text: str) -> str:
    prefixes = ["เพิ่ม", "ลบ", "เช็ค", "ตรวจสอบ", "เตือน", "ช่วย", "ขอให้", "อย่าลืม", "แก้ไข", "task", "งาน"]
    suffixes = ["ให้หน่อย", "หน่อย", "ด้วยนะ", "นะ", "สิ"]
    text = text.strip()
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    for s in suffixes:
        text = re.sub(rf"{s}$", "", text).strip()
    return text

def decorate_input(text: str):
    dt = extract_datetime(text)
    cleaned = clean_task_input(text)
    prompt = f"Format task: {cleaned}"
    inputs = decor_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = decor_model.generate(**inputs, max_length=64)
    decoded = decor_tokenizer.decode(outputs[0], skip_special_tokens=True)
    action = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+", "", decoded).strip()

    if dt:
        return {
            "task": action,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M")
        }
    else:
        return {"task": action, "date": "", "time": ""}

def classify_path(text: str):
    inputs = cls_path_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = cls_path_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
    max_index = int(torch.argmax(torch.tensor(probs)))
    return [max_index]

def classify_task(text: str, threshold: float = 0.5):
    inputs = task_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = task_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
    raw = [task_labels[i] for i, p in enumerate(probs) if p > threshold]
    return [r.lower() for r in raw]

# ====== DB hooks (demo; replace with your real DB) ======
TASK_DB = set()

def _db_key(payload):
    return (payload.get("task","").strip(), payload.get("date",""), payload.get("time",""))

def db_check(payload, intent):
    key = _db_key(payload)
    if intent == "add":
        if key in TASK_DB:
            return False, "รายการนี้มีอยู่แล้ว"
        return True, ""
    if intent in ("edit", "remove"):
        if key not in TASK_DB:
            return False, "ไม่พบงานในระบบ"
        return True, ""
    return True, ""

def db_apply_mutation(intent, payload):
    key = _db_key(payload)
    if intent == "add":
        TASK_DB.add(key); return True
    if intent == "remove":
        if key in TASK_DB:
            TASK_DB.remove(key); return True
        return False
    if intent == "edit":
        TASK_DB.add(key); return True
    return True

# ====== Core logic: per-phrase "check first" ======
FORCE_CHECK_BEFORE_MUTATION = True
MUTATION_INTENTS = {"add", "edit", "remove"}
INTENT_VERB = {"add": "เพิ่ม", "edit": "แก้ไข", "remove": "ลบ", "check": "ตรวจสอบ"}

def prioritize_intents(intents):
    priority = {"check": 0, "add": 1, "edit": 1, "remove": 1}
    return sorted(intents, key=lambda x: priority.get(x, 99))

def ensure_check_before_mutation(intents):
    intents = [i.lower() for i in intents]
    intents = list(dict.fromkeys(intents))
    if FORCE_CHECK_BEFORE_MUTATION and any(i in MUTATION_INTENTS for i in intents) and "check" not in intents:
        intents = ["check"] + intents
    return prioritize_intents(intents)

def split_input(text: str):
    parts = re.split(r"(?:และ|,|\s*\n+)", text)
    return [p.strip() for p in parts if p and p.strip()]

def process_input(text: str):
    path_labels = classify_path(text)
    path_names = ["task", "search", "fill_form", "exit_all", "exit_this"]

    # >>> เปลี่ยนโครงสร้างผลลัพธ์: ไม่ใส่ input_path อีกต่อไป
    result = {
        "text": text,             
        "decorated_input":"",    # จะไม่มีฟิลด์ intent ภายใน
        "message": ""
    }

    all_messages = []

    for idx in path_labels:
        path = path_names[idx]
        if path != "task":
            result["message"] = f"คำสั่งอยู่ในประเภท '{path}' ยังไม่รองรับรายละเอียดเพิ่มเติมครับ"
            continue

        phrases = split_input(text)
        decorated_payloads = []
        ordered_intents_global = []

        for phrase in phrases:
            intents = classify_task(phrase)
            if not intents:
                continue
            intents = ensure_check_before_mutation(intents)
            payload = decorate_input(phrase)
            decorated_payloads.append(payload)
            payload["intent"] = intents
            target_mutations = [i for i in intents if i in MUTATION_INTENTS]

            if "check" in intents and target_mutations:
                all_messages.append("ตรวจสอบงานให้ก่อนนะครับ")
                for mut in target_mutations:
                    ok, reason = db_check(payload, mut)
                    if not ok:
                        all_messages.append(f"ตรวจสอบแล้ว พบปัญหา: {reason}")
                    else:
                        applied = db_apply_mutation(mut, payload)
                        if applied:
                            all_messages.append(f"{INTENT_VERB[mut]}งานให้เรียบร้อยแล้วนะครับ")
                        else:
                            all_messages.append(f"{INTENT_VERB[mut]}ไม่สำเร็จ กรุณาลองอีกครั้ง")
            elif "check" in intents and not target_mutations:
                all_messages.append("ตรวจสอบงานให้ก่อนนะครับ")
            elif target_mutations:
                for mut in target_mutations:
                    applied = db_apply_mutation(mut, payload)
                    if applied:
                        all_messages.append(f"{INTENT_VERB[mut]}งานให้เรียบร้อยแล้วนะครับ")
                    else:
                        all_messages.append(f"{INTENT_VERB[mut]}ไม่สำเร็จ กรุณาลองอีกครั้ง")

            for it in intents:
                if it not in ordered_intents_global:
                    ordered_intents_global.append(it)

        # >>> ใส่เฉพาะ text + decorated (ไม่มี intent ภายใน)
        result["decorated_input"] ={
            "text": text,
            "decorated": decorated_payloads
        }

    result["message"] = " แล้ว ".join(all_messages) if all_messages else "ดำเนินการตามคำสั่งให้แล้วครับ"
    return result

# ====== FastAPI Endpoint ======
@app.post("/handle_input")
def handle_input(input: InputText):
    return process_input(input.text)

# ====== Run directly (uvicorn) ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
