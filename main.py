from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("RF_model.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
  return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Time_spent_Alone: float = Form(...),
    Stage_fear: str = Form(...),
    Social_event_attendance: float = Form(...),
    Going_outside: float = Form(...),
    Drained_after_socializing: str = Form(...),
    Friends_circle_size: float = Form(...),
    Post_frequency: float = Form(...)
):
  # Mapear variables categóricas a valores numéricos
  stage_fear_val = 1 if Stage_fear.lower() == "yes" else 0
  drained_val = 1 if Drained_after_socializing.lower() == "yes" else 0

  # Construir vector de características
  features = np.array([[
    Time_spent_Alone,
    stage_fear_val,
    Social_event_attendance,
    Going_outside,
    drained_val,
    Friends_circle_size,
    Post_frequency
  ]])

  prediction = model.predict(features)[0]

  if prediction == 0:
      result_text = "Eres una persona extrovertida"
      message = "Disfrutas estar con otras personas y participas activamente en actividades sociales."
  else:
      result_text = "Eres una persona introvertida"
      message = "Prefieres espacios tranquilos y disfrutas pasar tiempo a solas o en pequeños grupos."

  return templates.TemplateResponse("form.html", {
      "request": request,
      "result": result_text,
      "interpretation": message
  })