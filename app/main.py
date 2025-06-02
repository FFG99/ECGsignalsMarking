from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import mne
import os
import pickle
from typing import List, Dict
import logging
import warnings

from . import models, schemas
from .database import engine, get_db
from model_manager import ModelManager


warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="neurokit2.signal.signal_psd")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="neurokit2.hrv.hrv_time")

models.Base.metadata.create_all(bind=engine)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECG Signals Marking API",
    description="API для автоматической маркировки состояний по ЭКГ сигналам",
    version="1.0.0"
)

model, scaler, label_encoder, model_info = ModelManager.load_model()


@app.post("/records/upload", response_model=schemas.ECGRecord)
async def upload_record(file: UploadFile = File(...),
                        db: Session = Depends(get_db)) -> schemas.ECGRecord:
    """
    Загружает EDF файл с ЭКГ записью в базу данных.

    Args:
        file (UploadFile): EDF файл с записью ЭКГ
        db (Session): Сессия базы данных

    Returns:
        schemas.ECGRecord: Созданная запись ЭКГ

    Raises:
        HTTPException: При ошибке загрузки или обработки файла
    """
    logger.info(f"Processing upload request for file: {file.filename}")
    temp_file = f"temp_{file.filename}"

    try:
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.debug(f"Reading EDF file: {temp_file}")
        raw = mne.io.read_raw_edf(temp_file, preload=True)
        data = raw.get_data()

        logger.info(f"Successfully read EDF file with shape: {data.shape}")

        record = models.ECGRecord(
            filename=file.filename,
            data=pickle.dumps(data)
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(f"Successfully saved record with ID: {record.id}")
        return record

    except Exception as e:
        logger.error("Error processing file" +
                     f"{file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.debug(f"Removed temporary file: {temp_file}")


@app.get("/records/{record_id}/data")
async def get_ecg_data(record_id: int,
                       db: Session = Depends(get_db)) -> Dict[
                           str, List[List[float]]]:
    """
    Получает данные ЭКГ из базы данных по ID записи.

    Args:
        record_id (int): ID записи ЭКГ
        db (Session): Сессия базы данных

    Returns:
        Dict[str, List[List[float]]]: Словарь с данными ЭКГ

    Raises:
        HTTPException: Если запись не найдена или произошла ошибка
    """
    logger.info(f"Fetching ECG data for record ID: {record_id}")

    record = db.query(models.ECGRecord).\
        filter(models.ECGRecord.id == record_id).\
        first()

    if not record:
        logger.warning(f"Record not found with ID: {record_id}")
        raise HTTPException(status_code=404, detail="Record not found")

    try:
        data = pickle.loads(record.data)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return {"data": data.tolist()}

    except Exception as e:
        logger.error("Error loading data for record" +
                     f"{record_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records/", response_model=List[schemas.ECGRecord])
async def list_records(db: Session = Depends(get_db)) -> List[
                           schemas.ECGRecord]:
    """
    Получает список всех записей ЭКГ.

    Args:
        db (Session): Сессия базы данных

    Returns:
        List[schemas.ECGRecord]: Список записей ЭКГ
    """
    logger.info("Fetching list of all ECG records")
    records = db.query(models.ECGRecord).all()
    logger.info(f"Found {len(records)} records")
    return records


@app.delete("/records/{record_id}")
async def delete_record(record_id: int,
                        db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Удаляет запись ЭКГ из базы данных.

    Args:
        record_id (int): ID записи ЭКГ
        db (Session): Сессия базы данных

    Returns:
        Dict[str, str]: Сообщение об успешном удалении

    Raises:
        HTTPException: Если запись не найдена
    """
    logger.info(f"Deleting record with ID: {record_id}")

    record = db.query(models.ECGRecord).\
        filter(models.ECGRecord.id == record_id).\
        first()
    if not record:
        logger.warning(f"Record not found with ID: {record_id}")
        raise HTTPException(status_code=404, detail="Record not found")

    db.delete(record)
    db.commit()
    logger.info(f"Successfully deleted record with ID: {record_id}")
    return {"message": f"Record {record_id} deleted successfully"}


@app.post("/records/{record_id}/predict",
          response_model=schemas.PredictionResponse)
async def predict_record(
    record_id: int,
    db: Session = Depends(get_db)
) -> schemas.PredictionResponse:
    """
    Предсказывает состояния для записи ЭКГ.

    Args:
        record_id (int): ID записи ЭКГ
        db (Session): Сессия базы данных

    Returns:
        schemas.PredictionResponse: Предсказанные состояния

    Raises:
        HTTPException: Если запись не найдена или произошла ошибка
    """
    logger.info(f"Starting prediction for record ID: {record_id}")

    record = db.query(models.ECGRecord).\
        filter(models.ECGRecord.id == record_id).\
        first()
    if not record:
        logger.warning(f"Record not found with ID: {record_id}")
        raise HTTPException(status_code=404, detail="Record not found")

    try:
        data = pickle.loads(record.data)
        logger.info(f"Loaded data shape: {data.shape}")

        # Находим канал ЭКГ
        ecg_channel = None
        for i, channel_data in enumerate(data):
            if len(channel_data) > 0:
                ecg_channel = channel_data
                logger.info(f"Found ECG channel at index {i}" +
                            f"with length {len(channel_data)}")
                break

        if ecg_channel is None:
            logger.error("No valid ECG channel found in the data")
            raise HTTPException(status_code=500,
                                detail="No valid ECG channel found")

        # Получаем предсказания с объединением интервалов
        try:
            prediction_segments = ModelManager.predict_record(
                model, scaler, label_encoder, ecg_channel
            )
            logger.info("Generated" +
                        f"{len(prediction_segments)} prediction segments")
        except Exception as e:
            logger.error("Error in predict_record" +
                         f"{str(e)}", exc_info=True)
            raise HTTPException(status_code=500,
                                detail=f"Error in prediction: {str(e)}")

        segments = [
            schemas.PredictionSegment(
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                state=segment['prediction']
            )
            for segment in prediction_segments
        ]

        logger.info("Successfully completed prediction for record" +
                    f"{record_id}")
        return schemas.PredictionResponse(
            record_id=record_id,
            predictions=segments
        )

    except Exception as e:
        logger.error("Unexpected error in predict_record" +
                     f"{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
