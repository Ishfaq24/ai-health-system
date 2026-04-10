from fastapi import APIRouter
from src.schemas.disease_schema import DiseaseRequest
from src.predict.predict_disease import predict_disease
from src.schemas.heart_schema import HeartRequest
from src.predict.predict_heart import predict_heart

router = APIRouter()

@router.get("/")
def home():
    return {"message": "AI Health ML Service Running 🚀"}

@router.post("/predict-disease")
def predict(req: DiseaseRequest):
    result = predict_disease(req.symptoms)

    if "error" in result:
        return {
            "status": "error",
            "message": result["error"]
        }

    return {
        "status": "success",
        "data": result
    }


@router.post("/predict-heart-risk")
def predict_heart_api(req: HeartRequest):
    result = predict_heart(req.dict())

    return {
        "status": "success",
        "data": result
    }