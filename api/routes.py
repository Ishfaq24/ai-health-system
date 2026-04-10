from fastapi import APIRouter
from src.schemas.disease_schema import DiseaseRequest
from src.predict.predict_disease import predict_disease

router = APIRouter()

@router.get("/")
def home():
    return {"message": "AI Health ML Service Running 🚀"}

@router.post("/predict-disease")
def predict(req: DiseaseRequest):
    result = predict_disease(req.symptoms)
    return result