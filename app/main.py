from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import mne
import tempfile
import os
import pickle
from typing import List

from . import models, schemas
from .database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


@app.post("/upload/", response_model=schemas.ECGRecord)
async def upload_ecg_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.edf'):
        raise HTTPException(status_code=400,
                            detail="Only .edf files are allowed")
    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            raw = mne.io.read_raw_edf(tmp_path, preload=True)
            data = raw.get_data()
        finally:
            os.remove(tmp_path)

        serialized_data = pickle.dumps(data)

        db_record = models.ECGRecord(
            filename=file.filename,
            data=serialized_data
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return db_record

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records/{record_id}/data")
async def get_ecg_data(record_id: int, db: Session = Depends(get_db)):
    record = db.query(models.ECGRecord).\
        filter(models.ECGRecord.id == record_id).\
        first()

    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    try:
        data = pickle.loads(record.data)
        return {"data": data.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records/", response_model=List[schemas.ECGRecord])
async def list_records(db: Session = Depends(get_db)):
    records = db.query(models.ECGRecord).all()
    return records


@app.delete("/records/{record_id}")
async def delete_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(models.ECGRecord).filter(models.ECGRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    db.delete(record)
    db.commit()
    return {"message": f"Record {record_id} deleted successfully"}
