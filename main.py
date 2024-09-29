import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


from routers.predict import router
from routers.data import router_data
from config.config import ENV_VARIABLES

from dotenv import load_dotenv

app = FastAPI()
app.title = "Spam detection for Twilio use case"
app.version = "0.0.1" 

app.include_router(router)
app.include_router(router_data)

@app.get('/', tags=['home'])
def message():
    return HTMLResponse('<h1>Backend ML - Spam detection for Twilio use case</h1>')

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
