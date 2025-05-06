from pydantic import BaseModel, ConfigDict
from datetime import datetime


class EEGRecordBase(BaseModel):
    filename: str


class EEGRecordCreate(EEGRecordBase):
    pass


class EEGRecord(EEGRecordBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
