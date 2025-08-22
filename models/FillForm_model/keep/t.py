import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import re

model_dir = "t5_autofill_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

def extract_fields(input_text):
    prompt = f"Extract fields: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded = decoded.strip()
    decoded = re.sub(r",\s*}", "}", decoded)
    return decoded

fields_needed = ["Name", "Address", "Age", "ID"]

test_input = "1103900144378 50 ปี ชวัลญา อภิรักษธนากร 88/125 ซอยร่วมมิตร แขวงศาลาธรรมศพ เขตทวีวัฒนา กรุงเทพมหานคร 10120"
output = extract_fields(test_input)

try:
    result = json.loads(output)
except Exception as e:
    print("⚠️ Output is not valid JSON:", output)
    result = {}

# --- ตรวจสอบว่า value ในแต่ละ field อยู่ใน input จริงมั้ย ถ้าไม่ ให้ลบทิ้ง ---
def value_in_input(val, input_text):
    # ต้องเจอ val ทั้งหมดใน input ถึงจะถือว่า OK (ตัดเว้นวรรค)
    if not val or not val.strip():
        return False
    # สำหรับชื่อ ให้เช็คแต่ละคำในชื่อว่าต้องอยู่ใน input ทุกคำ (สำหรับภาษาไทย/อังกฤษ)
    return all(part in input_text for part in val.strip().split())

for field in list(result.keys()):
    # ลบ field ที่ค่าหาไม่เจอใน input
    if not value_in_input(result[field], test_input):
        del result[field]

missing = [field for field in fields_needed if field not in result or not str(result[field]).strip()]

print("Result:", result)
if missing:
    print("❗ ข้อมูลยังไม่ครบ กรุณากรอก:", ", ".join(missing))
else:
    print("✅ กรอกข้อมูลครบทุกช่องแล้ว!")
