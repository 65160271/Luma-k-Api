from fastapi import FastAPI
from predict_router import router

app = FastAPI()

# mount router
app.include_router(router, prefix="/model", tags=["predict"])
