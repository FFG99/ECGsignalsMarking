from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List


class ECGRecordBase(BaseModel):
    filename: str


class ECGRecordCreate(ECGRecordBase):
    pass


class ECGRecord(ECGRecordBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionSegment(BaseModel):
    start_time: float
    end_time: float
    state: str


class PredictionResponse(BaseModel):
    record_id: int
    predictions: List[PredictionSegment]
