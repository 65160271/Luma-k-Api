import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import re
import os

def run_autofill(input_text):
    model_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models","FillForm_model", "t5_autofill_model")
)
    tokenizer = T5Tokenizer.from_pretrained(model_dir,local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True)

    def extract_fields(text):
        prompt = f"Extract fields: {text}"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = decoded.strip()
        decoded = re.sub(r",\s*}", "}", decoded)
        return formatted_result 

    def extract_number(text):
        m = re.search(r"\d+", str(text))
        return m.group(0) if m else text

    def value_in_input(val, original_input):
        if not val or not str(val).strip():
            return False
        return all(part in original_input for part in str(val).strip().split())

    fields_needed = ["ID", "Name", "Surname", "Address", "Age"]

    output = extract_fields(input_text)

    try:
        result = json.loads(output)
    except Exception as e:
        print("⚠️ Output is not valid JSON:", output)
        result = {}

    for field in list(result.keys()):
        if not value_in_input(result[field], input_text):
            del result[field]
        elif field == "Age":
            result[field] = extract_number(result[field])
        elif field == "Address" and result[field] == result.get("ID", ""):
            del result[field]

    formatted_result = {field: result.get(field, "") for field in fields_needed}
    missing = [field for field, val in formatted_result.items() if not str(val).strip()]

    print("Result:", formatted_result)
    if missing:
        print("❗ ข้อมูลยังไม่ครบ กรุณากรอก:", ", ".join(missing))
    else:
        print("✅ กรอกข้อมูลครบทุกช่องแล้ว!")
