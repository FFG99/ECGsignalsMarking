from pydantic import BaseModel, ConfigDict
from datetime import datetime


class ECGRecordBase(BaseModel):
    filename: str


class ECGRecordCreate(ECGRecordBase):
    pass


class ECGRecord(ECGRecordBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
