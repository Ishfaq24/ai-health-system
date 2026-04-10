from pydantic import BaseModel
from typing import List

class DiseaseRequest(BaseModel):
    symptoms: List[str]