from beanie import Document
from pydantic.v1 import BaseModel


class PredictionModel(Document):
    image_id: str
    label: str
    confidence: float



class PredictionRequest(BaseModel):
    file: bytes
