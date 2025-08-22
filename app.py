# app.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os, time, re

# ===== ใช้ของเดิม (ไม่แก้ไฟล์เดิม) =====
from task import (
    process_input as task_process_input,
    classify_path as task_classify_path,  # ใช้โมเดลตัดสิน path สำรอง
)
from autofill_core import run_autofill
from search_llm import ask_with_cli_and_fallback
from search_core import search_google
from scrape_core import scrape_text
from llm_core import ask_llm_raw
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Unified Orchestrator (intent-first, form schema enforced, fallback=Search)")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

STATE: Dict[str, Any] = {"logs": []}

class ChatIn(BaseModel):
    text: str

# ---------- helpers ----------
def _now() -> float:
    return time.time()

def add_log(feature: str, role: str, text: str) -> None:
    STATE["logs"].append({"ts": _now(), "feature": feature, "role": role, "text": text})

def build_search_prompt(new_user_text: str, max_pairs: int = 3) -> str:
    hist = [e for e in STATE["logs"] if e["feature"] == "search"]
    pairs: List[str] = []
    u_buf: Optional[str] = None
    for e in hist[-(max_pairs * 2):]:
        if e["role"] == "user":
            u_buf = e["text"]
        elif e["role"] == "assistant" and u_buf is not None:
            pairs.append(f"ผู้ใช้: {u_buf}\nผู้ช่วย(ก่อนหน้า): {e['text']}")
            u_buf = None
    context = "\n\n".join(pairs)
    return f"บริบทก่อนหน้า:\n{context}\n\nคำถามใหม่: {new_user_text}" if context else new_user_text

def intents_from_task_out(out: Dict[str, Any]) -> List[str]:
    """ดึง intent จากผล task.process_input() รองรับทั้งรูปแบบใหม่/เก่า"""
    items: List[str] = []
    if isinstance(out.get("intent"), list):
        items.extend(out["intent"])

    di = out.get("decorated_input")
    if isinstance(di, dict):
        decorated = di.get("decorated", [])
        if isinstance(decorated, list):
            for obj in decorated:
                if isinstance(obj, dict) and isinstance(obj.get("intent"), list):
                    items.extend(obj["intent"])

    old = out.get("decorated_inputs")
    if isinstance(old, list):
        for obj in old:
            if isinstance(obj, dict) and isinstance(obj.get("intent"), list):
                items.extend(obj["intent"])

    return [str(x).strip().lower() for x in items]

def decide_feature(text: str, out: Dict[str, Any]) -> str:
    """
    เลือกเส้นทางการทำงาน:
    - ถ้าเป็นคำถาม/ค้นหา/มีปี ค.ศ. → บังคับไป google_search
    - ถ้า intent จาก task ชี้ชัด → ตามนั้น
    - ถ้าถูกเดาเป็น task แต่ payload ดูเหมือนวันที่/ตัวเลข → เปลี่ยนเป็น google_search
    - else → ตัวจำแนกสำรอง/คีย์เวิร์ด
    """
    low = (text or "").strip().lower()

    # คำบ่งชี้ว่าเป็นการค้นหา
    q_kw = ("ทาย","เดา","คาดการณ์","ใคร","อะไร","ยังไง","ยอดฮิต","นิยม","เทรนด์","trend","อันดับ","top","ในปี20","ปี 20","ปี20")
    search_kw = (
        "ค้นหา","หาข้อมูล","search","lookup","look up",
        "ข้อมูลของ","ข้อมูลเกี่ยวกับ","เกี่ยวกับ","คืออะไร","คือใคร",
        "review","รีวิว","ราคา","เปรียบเทียบ","เทียบ","meaning","แปล",
        "what","who","where","when","why","how"
    )

    # ถ้าขึ้นต้นด้วย "หา" หรือมี ? หรือมีคีย์เวิร์ดค้นหา → ไป search เว้นแต่เป็นคำสั่งจัดการงาน
    if low.startswith("หา") or "?" in low or any(k in low for k in search_kw):
        task_kw = ("เพิ่ม","แก้","แก้ไข","ลบ","นำออก","remove","edit","update","add","บันทึก")
        if not any(k in low for k in task_kw):
            return "google_search"

    # มีปี/คีย์เวิร์ดที่มักต้องค้นหา
    has_year = bool(re.search(r"(?:19|20)\d{2}", low))
    if has_year or any(k in low for k in q_kw):
        return "google_search"

    # intents จาก task
    its = intents_from_task_out(out)
    if any("googlesearch" in i for i in its): return "google_search"
    if any("search"       in i for i in its): return "search"
    if any("plan"         in i for i in its): return "plan"
    if any(("form" in i) or ("accidentreport" in i) for i in its): return "fill_form"

    feature_guess = "task" if any(i in ("task","add","check","edit","remove") for i in its) else ""

    # ถ้าเดาเป็น task แต่ payload เหมือนสแปมวันที่/เลข → ส่งไปค้นหา
    if feature_guess == "task":
        di = out.get("decorated_input")
        if isinstance(di, dict):
            dec = di.get("decorated", [])
            if isinstance(dec, list) and dec:
                task_text = ""
                first = dec[0]
                if isinstance(first, dict):
                    task_text = str(first.get("task") or first.get("text") or "")
                looks_like_date = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", task_text))
                digit_ratio = (sum(ch.isdigit() for ch in task_text) / max(len(task_text), 1))
                if looks_like_date or digit_ratio > 0.30:
                    return "google_search"
        return "task"

    # ตัวจำแนกสำรองจากโมเดล
    try:
        idxs = task_classify_path(text) or []
        mapping = ["task", "search", "fill_form", "exit_all", "exit_this"]
        if idxs and 0 <= idxs[0] < len(mapping):
            return mapping[idxs[0]]
    except Exception:
        pass

    # คีย์เวิร์ดฟอร์ม
    form_kw = ("form","แบบฟอร์ม","ฟอร์ม","กรอก","เติม","บันทึก","accident","accidentreport","case","vin","ทะเบียน","ทะเบียนรถ")
    if any(k in low for k in form_kw):
        return "fill_form"

    return "unknown"


# ---------- form normalizer ----------
FORM_TEMPLATE: Dict[str, Any] = {
    "form_1": {
        "case_id": "", "participant_id": "",
        "car_info": {
            "manufacturer": "", "model": "", "model_year": "", "vin": "",
            "ccm": "", "registration": "", "type": "", "engine": "",
            "gear": "", "power": "", "weight": "", "loading_weight": ""
        },
        "driver_info": {
            "name": "", "license_id": "", "dob": "", "gender": "",
            "phone": "", "email": "", "nationality": "",
            "address": {"subdistrict": "", "district": "", "province": "", "zipcode": ""}
        }
    },
    "form_2": {
        "accident_id": "",
        "location": {"subdistrict": "", "district": "", "province": "", "zipcode": ""},
        "datetime": "", "weather": "", "road": "", "environment": "", "cause": "", "detail": ""
    }
}

def _merge_template(template: Any, data: Any) -> Any:
    if isinstance(template, dict):
        out = {}
        for k, v in template.items():
            if isinstance(data, dict) and k in data:
                out[k] = _merge_template(v, data[k])
            else:
                out[k] = _merge_template(v, {}) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(template, list):
        if isinstance(data, list):
            return data
        return []
    return data if (data is not None and data != {}) else template

def normalize_fillform_output(run_result: Any, original_text: str) -> Dict[str, Any]:
    error = ""
    raw = original_text
    confidence: float = 0.0
    payload: Any = {}
    data = run_result
    if isinstance(run_result, (list, tuple)) and len(run_result) >= 1:
        data = run_result[0]
        if len(run_result) >= 2 and isinstance(run_result[1], (int, float)):
            confidence = float(run_result[1])
    if isinstance(data, dict):
        error = str(data.get("error", "")) if data.get("error") is not None else ""
        raw = str(data.get("raw", original_text))
        confidence = float(data.get("confidence", confidence or 0.0)) if isinstance(data.get("confidence"), (int, float)) else (confidence or 0.0)
        payload = data.get("form") or data
    elif isinstance(data, str):
        raw = data
    form_norm = _merge_template(FORM_TEMPLATE, payload if isinstance(payload, dict) else {})
    return {"error": error, "raw": raw, "form": form_norm, "confidence": confidence}

# ---------- search helpers ----------
def _extract_answer_and_sources(resp: Any) -> tuple[str, List[str]]:
    """รับได้ทั้ง str / dict / tuple(list) แล้วคืน (answer, [sources])"""
    answer = ""
    sources: List[str] = []
    if isinstance(resp, (list, tuple)):
        if len(resp) >= 1 and isinstance(resp[0], str):
            answer = resp[0]
        if len(resp) >= 2:
            s = resp[1]
            if isinstance(s, str):
                sources = [s]
            elif isinstance(s, list):
                sources = [str(x) for x in s if x]
        return answer, sources
    if isinstance(resp, dict):
        if isinstance(resp.get("answer"), str):
            answer = resp["answer"]
        elif isinstance(resp.get("text"), str):
            answer = resp["text"]
        if isinstance(resp.get("source"), str):
            sources = [resp["source"]]
        elif isinstance(resp.get("sources"), list):
            sources = [str(x) for x in resp["sources"] if x]
        return answer, sources
    if isinstance(resp, str):
        answer = resp
    return answer, sources

def _force_google_with_sources(user_text: str) -> tuple[str, List[str]]:
    """ยิง Google + สรุปเนื้อหา และคืน (summary, [sources]) อย่างน้อย 1 ลิงก์ถ้ามี"""
    try:
        links = search_google(user_text, os.environ.get("SERPAPI_API_KEY"), num_results=3) or []
    except Exception:
        links = []

    summary = ""
    primary = links[0] if links else ""
    try:
        page_text = scrape_text(primary) if primary else ""
        if page_text:
            summary = ask_llm_raw(f"สรุปสั้น 5 บรรทัดเป็นไทย:\n{page_text[:4000]}")
    except Exception:
        pass

    if not summary:
        try:
            raw = ask_with_cli_and_fallback(user_text)
        except TypeError:
            raw = ask_with_cli_and_fallback(user_text, os.environ.get("SERPAPI_API_KEY"))
        if isinstance(raw, dict):
            summary = str(raw.get("answer", "")) or ""
        elif isinstance(raw, str):
            summary = raw
        else:
            summary = str(raw)

    return (summary or "สรุปจากการค้นหาด้วย Google"), links[:3]

# ---------- single endpoint ----------
@app.post("/chat")
def chat(req: ChatIn):
    text = (req.text or "").strip()

    # วิเคราะห์ intent ก่อนเสมอ (ใช้ตรวจ exit ด้วย)
    task_out = task_process_input(text)
    intents = intents_from_task_out(task_out)

    # ---------- ถ้ามีคำสั่งออก (exit) ให้ปลดล็อกและจบที่นี่ ----------
    if any(i in ("exit_all", "exit_this") for i in intents):
        STATE["feature_lock"] = None
        msg = "ออกจากโหมดปัจจุบันแล้ว"
        return {
            "text": text,
            "decorated_input": {
                "decorated": [
                    {"text": text, "response": msg, "intent": ["Exit"]}
                ]
            },
            "message": msg
        }

    # ---------- ตัดสิน feature ปกติ ----------
    decided = decide_feature(text, task_out)

    # normalize: google_search ก็ถือเป็น search-mode เวลา lock
    def _to_lock_key(feat: str) -> str:
        return "search" if feat == "google_search" else feat

    # ---------- ใช้ feature lock ----------
    lock = STATE.get("feature_lock")
    if lock:
        # ถ้าผู้ใช้พิมพ์แนว "ค้นหา" ชัดเจน ให้ฝ่า lock ไป search ทันที
        if decided in ("search", "google_search") and lock != "search":
            feature = "search"
            STATE["feature_lock"] = "search"
        else:
            # คงโหมดเดิมตาม lock จนกว่าจะ exit
            feature = lock
    else:
        # ยังไม่ล็อก: ใช้โหมดที่ตัดสินได้ และตั้ง lock ให้โหมดหลัก
        feature = decided
        if feature in ("task", "search", "plan", "fill_form", "google_search"):
            STATE["feature_lock"] = _to_lock_key(feature)
            if feature == "google_search":
                feature = "search"   # run branch search จริง

    # ---------- helper คืนค่ามาตรฐาน ----------
    def std_response(message: str, decorated: Optional[List[Dict[str, Any]]] = None):
        return {
            "text": text,
            "decorated_input": {"decorated": decorated or []},
            "message": message
        }

    # ---------- ฟีเจอร์: task ----------
    if feature == "task":
        di = task_out.get("decorated_input") or {}
        decorated = di.get("decorated", []) if isinstance(di, dict) else []
        return std_response(task_out.get("message", ""), decorated)

    # ---------- ฟีเจอร์: search ----------
    if feature == "search":
        add_log("search", "user", text)
        prompt = build_search_prompt(text)
        try:
            raw = ask_with_cli_and_fallback(prompt)
        except TypeError:
            raw = ask_with_cli_and_fallback(prompt, os.environ.get("SERPAPI_API_KEY"))

        answer, sources = _extract_answer_and_sources(raw)
        unknown_flags = ("ไม่รู้", "ไม่มั่นใจ", "ไม่มีข้อมูล", "ไม่พบ", "ขออภัย", "ตอบไม่ได้")

        if (not answer) or any(f in str(answer) for f in unknown_flags):
            answer, sources = _force_google_with_sources(text)
            intent = ["Search", "GoogleSearch"]
        else:
            intent = ["Search"]

        add_log("search", "assistant", answer)
        return std_response(answer, [
            {"text": text, "response": answer, "sources": sources, "intent": intent}
        ])

    # ---------- ฟีเจอร์: plan ----------
    if feature == "plan":
        plan_text = ask_llm_raw(text)
        return std_response(plan_text, [
            {"text": text, "response": plan_text, "intent": ["Plan"]}
        ])

    # ---------- ฟีเจอร์: fill_form ----------
    if feature == "fill_form":
        try:
            result = run_autofill(text)
            norm = normalize_fillform_output(result, text)
            decorated = [{
                "text": text,
                "form": norm.get("form", {}),
                "error": norm.get("error", ""),
                "confidence": norm.get("confidence", 0.0),
                "intent": ["FillForm"]
            }]
            msg = "ประมวลผลฟอร์มเรียบร้อย" if not norm.get("error") else f"พบข้อผิดพลาด: {norm['error']}"
            return std_response(msg, decorated)
        except Exception as e:
            decorated = [{
                "text": text,
                "form": _merge_template(FORM_TEMPLATE, {}),
                "error": f"{type(e).__name__}: {e}",
                "confidence": 0.0,
                "intent": ["FillForm"]
            }]
            return std_response("พบข้อผิดพลาดระหว่างประมวลผลฟอร์ม", decorated)

    # ---------- อื่น ๆ ----------
    if STATE.get("feature_lock"):
        msg = f"อยู่ในโหมด '{STATE['feature_lock']}' — ใช้คำสั่ง exit เพื่อออกจากโหมดนี้"
        return std_response(msg, [{"text": text, "response": msg, "intent": ["Notice"]}])

    add_log("search", "user", text)
    answer, sources = _force_google_with_sources(text)
    add_log("search", "assistant", answer)
    return std_response(answer, [
        {"text": text, "response": answer, "sources": sources, "intent": ["Search"]}
    ])
