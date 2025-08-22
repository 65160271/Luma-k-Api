import json

input_file = "data/train.jsonl"    # ไฟล์เดิม (output เป็น Name เต็ม)
output_file = "output.jsonl"  # ไฟล์ใหม่ (output แยก Name/Surname)

def split_fullname(fullname):
    if not fullname or not isinstance(fullname, str):
        return "", ""
    parts = fullname.strip().split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    elif len(parts) == 1:
        return parts[0], ""
    else:
        return "", ""

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        # ถ้า "output" เป็น string json → แปลงก่อน
        out = obj["output"]
        if isinstance(out, str):
            out_dict = json.loads(out)
        else:
            out_dict = out
        # แยก Name/Surname เฉพาะถ้ามี Name
        if "Name" in out_dict and out_dict["Name"]:
            name, surname = split_fullname(out_dict["Name"])
            out_dict["Name"] = name
            out_dict["Surname"] = surname
        # เขียนกลับไปที่ "output" (string json)
        obj["output"] = json.dumps(out_dict, ensure_ascii=False)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ Done! Output file:", output_file)
