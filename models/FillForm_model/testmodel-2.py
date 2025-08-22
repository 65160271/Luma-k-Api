from fastapi import APIRouter
from pydantic import BaseModel
import torch, re, json, logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

router = APIRouter()

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FillFormPOC")

# โหลดโมเดลจาก path เดิม
model_path = "t5_autofill_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class PredictRequest(BaseModel):
    text: str

def safe_json_parse(text: str):
    logger.info(f"[CLEANER] Raw model output: {text}")
    fixed = {}

    # regex ดึงคู่ key:value
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text)
    logger.info(f"[CLEANER] Extracted pairs: {pairs}")

    for k, v in pairs:
        if k == "student_job_position" and "student_job_position" in v:
            last_val = v.split(":")[-1].strip().strip('"')
            logger.info(f"[CLEANER] Fixed {k} => {last_val}")
            fixed[k] = last_val
        else:
            fixed[k] = v

    if not fixed:
        return {"error": "Invalid JSON output", "raw": text}

    return fixed

def predict_with_confidence(input_text, max_length=512):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True
        )

    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    decoded_clean = decoded.lstrip("<extra_id_0>:").strip()
    logger.info(f"[MODEL] Decoded raw text: {decoded_clean}")

    # confidence
    scores = outputs.scores
    probs = [torch.softmax(score, dim=-1) for score in scores]
    token_ids = outputs.sequences[0][1:1+len(probs)]
    selected_probs = [float(prob[0, tid]) for prob, tid in zip(probs, token_ids)]
    avg_conf = sum(selected_probs) / len(selected_probs) if selected_probs else 0.0

    parsed_output = safe_json_parse(decoded_clean)
    parsed_output["confidence"] = avg_conf
    return parsed_output

@router.post("/predict")
async def predict(request: PredictRequest):
    logger.info(f"[REQUEST] Input text: {request.text}")
    result = predict_with_confidence(request.text)
    logger.info(f"[RESPONSE] Output: {result}")
    return result
