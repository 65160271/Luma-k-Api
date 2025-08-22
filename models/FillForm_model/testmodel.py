import json
import re
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from copy import deepcopy

router = APIRouter()

# --- โหลดโมเดลและ tokenizer ---
model_path = "t5_autofill_model"
tokenizer = MT5Tokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --- Request schema ---
class PredictRequest(BaseModel):
    text: str = None
    form_1: dict = {}
    form_2: dict = {}

# --- ฟอร์ม Template ---
FORM_TEMPLATE = {
    "form_1": {
        "case_id": "",
        "participant_id": "",
        "car_info": {
            "manufacturer": "",
            "model": "",
            "model_year": "",
            "vin": "",
            "ccm": "",
            "registration": "",
            "type": "",
            "engine": "",
            "gear": ""
        },
        "driver_info": {
            "name": "",
            "license_id": "",
            "dob": "",
            "gender": "",
            "phone": "",
            "email": "",
            "nationality": "",
            "address": {
                "subdistrict": "",
                "district": "",
                "province": "",
                "zipcode": ""
            }
        }
    },
    "form_2": {
        "accident_id": "",
        "location": {
            "subdistrict": "",
            "district": "",
            "province": "",
            "zipcode": ""
        },
        "datetime": "",
        "weather": "",
        "road": "",
        "environment": "",
        "cause": "",
        "detail": ""
    }
}

def merge_with_template(decoded: str):
    """พยายาม parse decoded string และเติม field ที่ขาด"""
    try:
        parsed = json.loads(decoded)
    except Exception:
        # ถ้า parse ไม่ได้เลย → return template เปล่า พร้อม raw string
        return deepcopy(FORM_TEMPLATE), False

    def fill(template, data):
        for k, v in template.items():
            if isinstance(v, dict):
                template[k] = fill(v, data.get(k, {}))
            else:
                template[k] = data.get(k, "")
        return template

    filled = fill(deepcopy(FORM_TEMPLATE), parsed)
    return filled, True

# --- fill_missing ---
def fill_missing(user_data: dict, template: dict):
    result = deepcopy(template)
    for key, val in user_data.items():
        if isinstance(val, dict) and key in result:
            result[key] = fill_missing(val, result[key])
        else:
            result[key] = val
    return result

# --- serialize_record ---
def serialize_record(form_1, form_2):
    car = form_1.get("car_info", {})
    driver = form_2.get("driver_info", {})
    location = form_2.get("location", {})
    input_text = (
        f"Case {form_1.get('case_id', '')}, Accident {form_2.get('accident_id', '')}. "
        f"Driver {driver.get('name', '')} {driver.get('surname', '')}, "
        f"age {driver.get('age', '')}, injury {driver.get('injury', '')}. "
        f"Car {car.get('manufacturer', '')} {car.get('model', '')}, "
        f"year {car.get('model_year', '')}, type {car.get('type', '')}. "
        f"Location: {location.get('province', '')}, {location.get('district', '')}, {location.get('subdistrict', '')}."
    )
    print('[DEBUG] serialized input_text =', repr(input_text))
    return input_text

# --- parser สำหรับทุกเคส ---
def parse_text_to_form(text: str) -> dict:
    data = {
        "case_id": "",
        "participant_id": "",
        "car_info": {},
        "usage": "",
        "occupants": [],
        "accident_id": "",
        "location": {},
        "driver_info": {},
        "vehicle_status": {}
    }
    tokens = text.strip().split()

    for i, token in enumerate(tokens):
        token_l = token.lower()

        # form_1
        if re.match(r"^co\d+$", token_l):
            data["case_id"] = token
        elif token_l.startswith("p") and token_l[1:].isdigit():
            data["participant_id"] = token
        elif token_l in ["honda", "toyota", "mazda", "nissan", "ford", "bmw", "benz"]:
            data["car_info"]["manufacturer"] = token
        elif token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100:
            data["car_info"]["model_year"] = token
        elif len(token) == 17 and token.isalnum():
            data["car_info"]["vin"] = token
        elif token_l.endswith("cc"):
            data["car_info"]["ccm"] = token_l.replace("cc", "")
        elif token_l.startswith("reg"):
            data["car_info"]["registration"] = token[3:]
        elif token_l in ["sedan", "suv", "truck", "van", "pickup"]:
            data["car_info"]["type"] = token_l
        elif token_l in ["petrol", "diesel", "ev", "hybrid"]:
            data["car_info"]["engine"] = token_l
        elif token_l in ["yes", "no"]:
            data["car_info"]["hybrid"] = token_l
        elif token_l.endswith("kw"):
            data["car_info"]["power"] = token_l.replace("kw", "")
        elif token_l.endswith("kg"):
            num = token_l.replace("kg", "")
            if not data["car_info"].get("weight"):
                data["car_info"]["weight"] = num
            else:
                data["car_info"]["loading_weight"] = num
        elif token_l in ["personal", "commercial", "government"]:
            data["usage"] = token_l

        # form_2
        elif re.match(r"^ac\d+$", token_l):
            data["accident_id"] = token
        elif token_l in ["bangkok", "chiangmai", "chonburi"]:
            data["location"]["province"] = token
        elif re.match(r"^\d{5}$", token):
            data["location"]["zipcode"] = token
        elif token_l.endswith("road") or token_l.endswith("rd"):
            data["location"]["street"] = token
        elif token_l.endswith("district"):
            data["location"]["district"] = token
        elif token_l.endswith("subdistrict"):
            data["location"]["subdistrict"] = token
        elif token_l.istitle() and i + 1 < len(tokens) and tokens[i + 1].istitle():
            data["driver_info"]["name"] = token
            data["driver_info"]["surname"] = tokens[i + 1]
        elif token.isdigit() and 15 <= int(token) <= 100:
            data["driver_info"]["age"] = token
        elif token_l in ["male", "female"]:
            data["driver_info"]["sex"] = token_l
        elif re.match(r"^[a-z]{2}\d{6}$", token_l):
            data["driver_info"]["license_no"] = token
        elif token_l in ["injured", "dead", "safe"]:
            data["driver_info"]["injury"] = token_l
        elif token_l.startswith("note"):
            data["driver_info"]["notes"] = token[4:]
        elif token_l in ["bumper", "door", "hood", "trunk"]:
            data["vehicle_status"]["damaged_parts"] = token_l
        elif token_l in ["repair", "replace", "totaled"]:
            data["vehicle_status"]["repair_needed"] = token_l
        elif token_l in ["viriyah", "msig", "tokio", "axa", "thanachart"]:
            data["vehicle_status"]["insurance_company"] = token
        elif re.match(r"^[A-Z]{2}\d{6}$", token):
            data["vehicle_status"]["policy_no"] = token

    return data

# --- predict endpoint ---
@router.post("/model/predict")
def predict(req: PredictRequest):
    decoded = ""
    form = deepcopy(FORM_TEMPLATE)

    try:
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        # decode ทั้งแบบ clean และแบบ raw
        decoded_raw = tokenizer.decode(outputs[0], skip_special_tokens=False).strip()
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("[DEBUG] decoded_raw:", repr(decoded_raw))
        print("[DEBUG] decoded_clean:", repr(decoded))

        # fallback ถ้า decoded ว่าง
        if not decoded and decoded_raw:
            decoded = decoded_raw

        # ถ้ายังว่าง → fallback ไป parse_text_to_form
        if not decoded:
            parsed_fallback = parse_text_to_form(req.text)
            return {
                "error": "fallback to regex parser",
                "raw": req.text,
                "form": fill_missing(parsed_fallback, FORM_TEMPLATE),
                "confidence": 0.0
            }

        form, valid = merge_with_template(decoded)

        return {
            "error": "" if valid else "incomplete form, fallback to template",
            "raw": decoded,
            "form": form,
            "confidence": float(len(decoded) / 512)
        }

    except Exception as e:
        # สุดท้ายก็ยังคืน decoded หรือใช้ regex parser
        parsed_fallback = parse_text_to_form(req.text)
        return {
            "error": f"parse error: {str(e)}",
            "raw": decoded if decoded else req.text,
            "form": fill_missing(parsed_fallback, FORM_TEMPLATE),
            "confidence": 0.0
        }




