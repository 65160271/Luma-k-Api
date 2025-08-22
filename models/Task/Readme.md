# To run the demo 
uvicorn app:app --reload

# Text Input 
ex. เพิ่มนัดประชุมพรุ่งนี้ตอนบ่ายสอง
# Example Response
{
  "text": "เพิ่มนัดประชุมพรุ่งนี้ตอนบ่ายสอง",
  "intent": [
    "Add"
  ],
  "decorated_input": {
    "detail": "นัดประชุม",
    "date": "2025-07-22",
    "time": "14:00"
  }
}
$