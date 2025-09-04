# app.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
import os, time, re

# ===== external modules (ของเดิม) =====
from task import (
    process_input as task_process_input,
    classify_path as task_classify_path,  # classifier สำรอง
)
from autofill_core import run_autofill
from search_llm import ask_with_cli_and_fallback
from search_core import search_google
from scrape_core import scrape_text
from llm_core import ask_llm_raw
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Unified Orchestrator (feature-lock + interactive fill_form)")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global STATE =====
STATE: Dict[str, Any] = {
    "logs": [],
    "feature_lock": None,              # "task" | "search" | "plan" | "fill_form" | None
    "fill_form_session": None,         # {"form": dict, "missing": [...], "filled": [...]}
    "task_session": None,
    "search_session": None,
}

# ===== Models =====
class ChatIn(BaseModel):
    text: str

# ---------- helpers: time/log ----------
def _now() -> float: return time.time()
def add_log(feature: str, role: str, text: str) -> None:
    STATE["logs"].append({"ts": _now(), "feature": feature, "role": role, "text": text})

# ---------- helpers: search context ----------
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

# ---------- helpers: intents ----------
def intents_from_task_out(out: Dict[str, Any]) -> List[str]:
    items: List[str] = []
    if isinstance(out.get("intent"), list): items.extend(out["intent"])
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

def _intent_bucket(name: str) -> str:
    n = (name or "").strip().lower()
    if "form" in n: return "fill_form"
    if "googlesearch" in n or "search" in n: return "search"
    if "plan" in n: return "plan"
    if n in ("task","add","check","edit","remove"): return "task"
    return ""

def detect_multi_feature_request(intents: List[str]) -> List[str]:
    buckets: List[str] = []
    for it in intents:
        b = _intent_bucket(it)
        if b and b not in buckets:
            buckets.append(b)
    return buckets

# ---------- helpers: switch / exit ----------
def parse_switch_command(text: str) -> Tuple[str, Optional[str]]:
    t = (text or "").strip().lower()
    if re.search(r"(ออก|จบ|พอแล้ว|หยุด|เลิก|end|exit)\b", t):
        return "exit", None
    if re.search(r"(ไป(ที่)?|สลับ|เปลี่ยน|switch)\s*(โหมด)?\s*(search|ค้นหา)", t):
        return "switch", "search"
    if re.search(r"(ไป(ที่)?|สลับ|เปลี่ยน|switch)\s*(โหมด)?\s*(task|งาน|ทาสก์)", t):
        return "switch", "task"
    if re.search(r"(ไป(ที่)?|สลับ|เปลี่ยน|switch)\s*(โหมด)?\s*(form|ฟอร์ม|กรอกฟอร์ม|fill\s*form)", t):
        return "switch", "fill_form"
    if re.search(r"(ไปทำอย่างอื่น|อย่างอื่น)", t):
        return "exit", None
    return "", None

# ---------- helpers: feature label ----------
def feature_badge(feature: str) -> str:
    mapping = {
        "task": "TASK",
        "search": "SEARCH",
        "plan": "PLAN",
        "fill_form": "FILL_FORM",
        "unknown": "UNKNOWN"
    }
    return f"[โหมด: {mapping.get(feature, feature).upper()}] "

# ---------- decide feature ----------
def decide_feature(text: str, out: Dict[str, Any]) -> str:
    low = (text or "").strip().lower()

    # เข้า fill_form เมื่อสั่งชัดเจนเท่านั้น
    explicit_fill = bool(re.search(r"(ช่วย)?\s*(กรอก|เติม)\s*(แบบ)?ฟอร์ม", low)) \
        or any(k in low for k in ("กรอกแบบฟอร์ม","เริ่มกรอกฟอร์ม","เริ่มฟอร์ม","เปิดฟอร์ม","กรอกข้อมูล","fill form","fillform","auto fill","autofill"))
    if explicit_fill: return "fill_form"

    its = intents_from_task_out(out)
    if any(("form" in i) or ("fill_form" in i) or ("accidentreport" in i) for i in its): return "fill_form"
    if any("plan" in i for i in its): return "plan"
    if any("search" in i for i in its) or any("googlesearch" in i for i in its): return "search"

    if low.startswith("หา") or "?" in low: return "search"
    if any(i in ("task","add","check","edit","remove") for i in its): return "task"

    try:
        idxs = task_classify_path(text) or []
        mapping = ["task", "search", "fill_form", "exit_all", "exit_this"]
        if idxs and 0 <= idxs[0] < len(mapping): return mapping[idxs[0]]
    except Exception:
        pass

    return "unknown"

# ---------- fill-form session utils ----------
def _deep_merge_form_keep_existing(base: Any, new: Any) -> Any:
    # deep-merge: ถ้า new เป็น "" จะไม่ทับของเดิม
    if isinstance(base, dict) and isinstance(new, dict):
        out = {}
        keys = set(base.keys()) | set(new.keys())
        for k in keys:
            if k in base and k in new:
                out[k] = _deep_merge_form_keep_existing(base[k], new[k])
            elif k in new:
                out[k] = new[k]
            else:
                out[k] = base[k]
        return out
    if isinstance(new, str):
        return new if new.strip() != "" else (base if isinstance(base, str) else new)
    if isinstance(new, (int, float)) and new is not None:
        return new
    if isinstance(new, list):
        return new
    return new if new is not None else base

def _collect_missing_paths(obj: Any, prefix: List[str] | None = None) -> List[str]:
    prefix = prefix or []
    miss: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            miss.extend(_collect_missing_paths(v, prefix + [k]))
        return miss
    if isinstance(obj, str) and obj.strip() == "":
        miss.append(".".join(prefix))
    return miss

def _humanize_field(path: str) -> str:
    labels = {
        "form_1.case_id": "รหัสเคส",
        "form_1.participant_id": "รหัสผู้เกี่ยวข้อง",

        "form_1.car_info.manufacturer": "ยี่ห้อรถ",
        "form_1.car_info.model": "รุ่นรถ",
        "form_1.car_info.model_year": "ปีรุ่น",
        "form_1.car_info.vin": "VIN",
        "form_1.car_info.registration": "ทะเบียนรถ",
        "form_1.car_info.type": "ประเภทรถ",
        "form_1.car_info.engine": "เครื่องยนต์",
        "form_1.car_info.gear": "เกียร์",
        "form_1.car_info.power": "แรงม้า",
        "form_1.car_info.weight": "น้ำหนักรถ",
        "form_1.car_info.loading_weight": "น้ำหนักบรรทุก",

        "form_1.driver_info.name": "ชื่อผู้ขับ",
        "form_1.driver_info.license_id": "ใบขับขี่",
        "form_1.driver_info.dob": "วันเกิดผู้ขับ",
        "form_1.driver_info.gender": "เพศผู้ขับ",
        "form_1.driver_info.phone": "โทรศัพท์",
        "form_1.driver_info.email": "อีเมล",
        "form_1.driver_info.nationality": "สัญชาติ",
        "form_1.driver_info.address.subdistrict": "ตำบล/แขวง",
        "form_1.driver_info.address.district": "อำเภอ/เขต",
        "form_1.driver_info.address.province": "จังหวัด",
        "form_1.driver_info.address.zipcode": "รหัสไปรษณีย์",

        "form_2.accident_id": "รหัสอุบัติเหตุ",
        "form_2.datetime": "วันเวลาเกิดเหตุ",
        "form_2.location.subdistrict": "จุดเกิดเหตุ-ตำบล",
        "form_2.location.district": "จุดเกิดเหตุ-อำเภอ",
        "form_2.location.province": "จุดเกิดเหตุ-จังหวัด",
        "form_2.location.zipcode": "จุดเกิดเหตุ-รหัสไปรษณีย์",
        "form_2.weather": "สภาพอากาศ",
        "form_2.road": "สภาพถนน",
        "form_2.environment": "สภาพแวดล้อม",
        "form_2.cause": "สาเหตุ",
        "form_2.detail": "รายละเอียดเหตุการณ์",
    }
    return labels.get(path, path)

def _pretty_missing(missing_paths: List[str], limit: int = 6) -> str:
    if not missing_paths:
        return "ไม่มีแล้ว"
    labels = [_humanize_field(p) for p in missing_paths[:limit]]
    tail = " ฯลฯ" if len(missing_paths) > limit else ""
    return ", ".join(labels) + tail

def _ask_prompt_for_field(path: str) -> str:
    prompts = {
        "form_1.case_id": "รหัสเคสคืออะไรครับ?",
        "form_1.participant_id": "รหัสผู้เกี่ยวข้องคืออะไรครับ?",

        "form_1.car_info.manufacturer": "รถยี่ห้ออะไรครับ?",
        "form_1.car_info.model": "รุ่นอะไรครับ?",
        "form_1.car_info.model_year": "ปีรุ่น (ค.ศ.) เท่าไหร่ครับ?",
        "form_1.car_info.vin": "หมายเลขตัวถัง (VIN) 17 ตัว คืออะไรครับ?",
        "form_1.car_info.registration": "ทะเบียนรถอะไรครับ?",
        "form_1.car_info.type": "ประเภทรถคืออะไรครับ (เช่น เก๋ง, กระบะ)?",
        "form_1.car_info.engine": "เครื่องยนต์รุ่น/รหัสอะไรครับ (ถ้ามี)?",
        "form_1.car_info.gear": "เกียร์อะไรครับ (ธรรมดา/อัตโนมัติ)?",

        "form_1.driver_info.name": "ชื่อ-นามสกุลผู้ขับคืออะไรครับ?",
        "form_1.driver_info.license_id": "เลขใบขับขี่คืออะไรครับ?",
        "form_1.driver_info.dob": "วันเกิดผู้ขับ (YYYY-MM-DD) คือวันที่เท่าไหร่ครับ?",
        "form_1.driver_info.gender": "เพศของผู้ขับคืออะไรครับ?",
        "form_1.driver_info.phone": "เบอร์โทรผู้ขับคืออะไรครับ?",
        "form_1.driver_info.email": "อีเมลผู้ขับคืออะไรครับ?",
        "form_1.driver_info.address.subdistrict": "ที่อยู่ผู้ขับ: ตำบล/แขวง อะไรครับ?",
        "form_1.driver_info.address.district": "ที่อยู่ผู้ขับ: อำเภอ/เขต อะไรครับ?",
        "form_1.driver_info.address.province": "ที่อยู่ผู้ขับ: จังหวัดอะไรครับ?",
        "form_1.driver_info.address.zipcode": "ที่อยู่ผู้ขับ: รหัสไปรษณีย์เท่าไหร่ครับ?",

        "form_2.accident_id": "รหัสอุบัติเหตุคืออะไรครับ?",
        "form_2.datetime": "เกิดเหตุวันเวลาไหนครับ? (เช่น 2025-01-30 14:30)",
        "form_2.location.subdistrict": "จุดเกิดเหตุ: ตำบล/แขวง อะไรครับ?",
        "form_2.location.district": "จุดเกิดเหตุ: อำเภอ/เขต อะไรครับ?",
        "form_2.location.province": "จุดเกิดเหตุ: จังหวัดอะไรครับ?",
        "form_2.location.zipcode": "จุดเกิดเหตุ: รหัสไปรษณีย์เท่าไหร่ครับ?",
        "form_2.weather": "ขณะเกิดเหตุสภาพอากาศเป็นอย่างไรครับ?",
        "form_2.road": "สภาพถนนเป็นอย่างไรครับ?",
        "form_2.environment": "สภาพแวดล้อมโดยรวมเป็นอย่างไรครับ?",
        "form_2.cause": "คาดว่าสาเหตุเกิดจากอะไรครับ?",
        "form_2.detail": "ช่วยเล่าเหตุการณ์โดยย่อหน่อยครับ",
    }
    return prompts.get(path, f"ขอข้อมูล **{_humanize_field(path)}** หน่อยครับ?")

# ---------- search helpers ----------
def _extract_answer_and_sources(resp: Any) -> tuple[str, List[str]]:
    answer = ""; sources: List[str] = []
    if isinstance(resp, (list, tuple)):
        if len(resp) >= 1 and isinstance(resp[0], str): answer = resp[0]
        if len(resp) >= 2:
            s = resp[1]
            if isinstance(s, str): sources = [s]
            elif isinstance(s, list): sources = [str(x) for x in s if x]
        return answer, sources
    if isinstance(resp, dict):
        if isinstance(resp.get("answer"), str): answer = resp["answer"]
        elif isinstance(resp.get("text"), str):  answer = resp["text"]
        if isinstance(resp.get("source"), str):  sources = [resp["source"]]
        elif isinstance(resp.get("sources"), list): sources = [str(x) for x in resp["sources"] if x]
        return answer, sources
    if isinstance(resp, str): answer = resp
    return answer, sources

def _force_google_with_sources(user_text: str) -> tuple[str, List[str]]:
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
        if isinstance(raw, dict): summary = str(raw.get("answer", "")) or ""
        elif isinstance(raw, str): summary = raw
        else: summary = str(raw)
    return (summary or "สรุปจากการค้นหาด้วย Google"), links[:3]

# ---------- main endpoint ----------
@app.post("/chat")
def chat(req: ChatIn):
    text = (req.text or "").strip()

    # (A) สั่งออก/สลับโหมด (มีผลกับ lock)
    action, target = parse_switch_command(text)
    if action == "exit":
        STATE["feature_lock"] = None
        STATE["fill_form_session"] = None
        msg = feature_badge("unknown") + "ออกจากโหมดปัจจุบันแล้ว"
        return {"text": text, "decorated_input": {"decorated": [
            {"text": text, "response": msg, "intent": ["Exit"]}
        ]}, "message": msg}
    elif action == "switch":
        STATE["feature_lock"] = target
        if target != "fill_form": STATE["fill_form_session"] = None
        # ไปทำตามโหมดใหม่ด้านล่าง

    # (B) วิเคราะห์ intent
    task_out = task_process_input(text)
    intents = intents_from_task_out(task_out)

    # (C) Guard: multi-feature (ยังไม่มี lock) → ให้ผู้ใช้เลือกก่อน
    if not STATE.get("feature_lock"):
        buckets = detect_multi_feature_request(intents)
        if len(buckets) > 1:
            msg = feature_badge("unknown") + (
                "ขอทำทีละฟีเจอร์นะครับ ตอนนี้เห็นว่ามีหลายอย่าง: "
                + ", ".join(buckets)
                + " — โปรดพิมพ์ 'ไปที่ค้นหา' / 'ไปที่ task' / 'ไปที่ฟอร์ม' หรือ 'ออก'"
            )
            return {
                "text": text,
                "decorated_input": {"decorated": [
                    {"text": text, "response": msg, "intent": ["MultiIntentConflict"], "intents_detected": buckets}
                ]},
                "message": msg
            }

    # (D) เลือก feature ถ้ายังไม่มี lock
    decided = decide_feature(text, task_out)

    # (E) FEATURE LOCK (เข้ม)
    lock = STATE.get("feature_lock")
    if lock:
        feature = lock
    else:
        feature = decided
        if feature in ("task", "search", "plan", "fill_form"):
            STATE["feature_lock"] = feature

    # helper: ส่งข้อความพร้อม badge โหมด
    def std_response(feature_name: str, message: str, decorated: Optional[List[Dict[str, Any]]] = None):
        return {
            "text": text,
            "decorated_input": {"decorated": decorated or []},
            "message": feature_badge(feature_name) + message
        }

    # ===== TASK (lock เข้ม) =====
    if feature == "task":
        di = task_out.get("decorated_input") or {}
        decorated = di.get("decorated", []) if isinstance(di, dict) else []
        msg = task_out.get("message", "บันทึกแล้ว") + " (พิมพ์ 'ออก' เพื่อไปทำอย่างอื่น)"
        return std_response("task", msg, decorated)

    # ===== SEARCH (lock เข้ม) =====
    if feature == "search":
        add_log("search", "user", text)
        prompt = build_search_prompt(text)
        try:
            raw = ask_with_cli_and_fallback(prompt)
        except TypeError:
            raw = ask_with_cli_and_fallback(prompt, os.environ.get("SERPAPI_API_KEY"))
        answer, sources = _extract_answer_and_sources(raw)
        unknown_flags = ("ไม่รู้","ไม่มั่นใจ","ไม่มีข้อมูล","ไม่พบ","ขออภัย","ตอบไม่ได้")
        if (not answer) or any(f in str(answer) for f in unknown_flags):
            answer, sources = _force_google_with_sources(text)
            intent = ["Search","GoogleSearch"]
        else:
            intent = ["Search"]
        add_log("search", "assistant", answer)
        resp = answer + "\n\n(พิมพ์ 'ออก' เพื่อไปทำอย่างอื่น)"
        return std_response("search", resp, [
            {"text": text, "response": answer, "sources": sources, "intent": intent}
        ])

    # ===== PLAN (option) =====
    if feature == "plan":
        plan_text = ask_llm_raw(text)
        resp = (plan_text or "วางแผนแล้ว") + "\n\n(พิมพ์ 'ออก' เพื่อไปทำอย่างอื่น)"
        return std_response("plan", resp, [
            {"text": text, "response": plan_text, "intent": ["Plan"]}
        ])

    # ===== FILL_FORM (interactive + lock เข้ม) =====
    if feature == "fill_form":
        af = run_autofill(text)  # คาดหวัง: {"form": {...}, "missing_fields": [...], "filled_fields": [...], "error": "", "confidence": 0.x}
        new_form = af.get("form", {})
        err = af.get("error", "")

        sess = STATE.get("fill_form_session") or {}
        cur_form = sess.get("form") or {}
        merged = _deep_merge_form_keep_existing(cur_form, new_form)

        missing = _collect_missing_paths(merged)
        filled: List[str] = []

        STATE["fill_form_session"] = {"form": merged, "missing": missing, "filled": filled}

        if err:
            msg = f"เกิดข้อผิดพลาดระหว่างเติมฟอร์ม: {err}\n(พิมพ์ 'ออก' เพื่อไปทำอย่างอื่น)"
        else:
            if missing:
                nice_list = _pretty_missing(missing)
                ask = _ask_prompt_for_field(missing[0])
                msg = (
                    "รับทราบครับ ผมเติมข้อมูลที่มีให้แล้ว 👍\n"
                    f"ตอนนี้ยังขาด: {nice_list}\n"
                    f"{ask}\n"
                    "(พิมพ์ 'ออก' เพื่อจบโหมดฟอร์มหรือสลับไปทำอย่างอื่นได้ทุกเมื่อ)"
                )
            else:
                msg = "เรียบร้อยครับ ✅ ข้อมูลครบทุกช่องแล้ว!\n(พิมพ์ 'ออก' เพื่อย้ายไปทำอย่างอื่น)"

        decorated = [{
            "text": text,
            "form": merged,
            "error": err,
            "confidence": af.get("confidence", 0.0),
            "intent": ["FillForm"],
            "missing_fields": missing,
            "filled_fields": filled,
            "notes": af.get("notes", "")
        }]
        return std_response("fill_form", msg, decorated)

    # ===== ถ้ามี lock แต่ไม่เข้าเงื่อนไขใด =====
    if STATE.get("feature_lock"):
        msg = f"อยู่ในโหมด '{STATE['feature_lock']}' — พิมพ์ 'ออก' เพื่อจบโหมดนี้หรือ 'ไปที่ <search/task/form>' เพื่อสลับ"
        return std_response(STATE["feature_lock"], msg, [{"text": text, "response": msg, "intent": ["Notice"]}])

    # ===== default: search + lock =====
    add_log("search", "user", text)
    answer, sources = _force_google_with_sources(text)
    add_log("search", "assistant", answer)
    if not STATE.get("feature_lock"): STATE["feature_lock"] = "search"
    resp = answer + "\n\n(พิมพ์ 'ออก' เพื่อไปทำอย่างอื่น)"
    return std_response("search", resp, [
        {"text": text, "response": answer, "sources": sources, "intent": ["Search"]}
    ])
