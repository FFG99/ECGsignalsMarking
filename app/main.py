from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import mne
import tempfile
import os
import pickle
from typing import List
import numpy as np
from pathlib import Path
import logging

from . import models, schemas
from .database import engine, get_db
from model_manager import ModelManager
from features.eeg_processor import EEGProcessor
import neurokit2 as nk
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2.signal.signal_psd")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2.hrv.hrv_time")

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

model, scaler, label_encoder, model_info = ModelManager.load_model() # Загрузка модели при старте

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(signal: np.ndarray, sampling_rate: int = 1000) -> np.ndarray:
    processor = EEGProcessor(sampling_rate=sampling_rate)
    
    segments = processor.segment_signal(signal, overlap=0.9)
    
    features_list = []
    for segment in segments:
        r_peaks = nk.ecg_findpeaks(segment, sampling_rate=sampling_rate)
        rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate
        
        if len(rr_intervals) > 0:
            features = processor.calculate_features(segment, rr_intervals)
            feature_array = np.array([features[feature] for feature in [
                'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50',
                'LF_power', 'HF_power', 'LF_HF_ratio'
            ]])
            features_list.append(feature_array)
    
    return np.array(features_list)


@app.post("/records/upload", response_model=schemas.ECGRecord)
async def upload_record(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        raw = mne.io.read_raw_edf(temp_file, preload=True)
        data = raw.get_data()

        record = models.ECGRecord(
            filename=file.filename,
            data=pickle.dumps(data)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        os.remove(temp_file)

        return record

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
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


@app.post("/records/{record_id}/predict", response_model=schemas.PredictionResponse)
async def predict_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(models.ECGRecord).filter(models.ECGRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    try:
        data = pickle.loads(record.data)
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Находим канал ЭКГ
        ecg_channel = None
        for i, channel_data in enumerate(data):
            if len(channel_data) > 0:
                ecg_channel = channel_data
                logger.info(f"Found ECG channel at index {i} with length {len(channel_data)}")
                break
        
        if ecg_channel is None:
            logger.error("No valid ECG channel found in the data")
            raise HTTPException(status_code=500, detail="No valid ECG channel found")
        
        # Получаем предсказания с объединением интервалов
        try:
            prediction_segments = ModelManager.predict_record(
                model, scaler, label_encoder, ecg_channel
            )
            logger.info(f"Generated {len(prediction_segments)} prediction segments")
        except Exception as e:
            logger.error(f"Error in predict_record: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
        
        # Преобразуем в формат ответа
        segments = [
            schemas.PredictionSegment(
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                state=segment['prediction']
            )
            for segment in prediction_segments
        ]
        
        return schemas.PredictionResponse(
            record_id=record_id,
            predictions=segments
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_record: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
