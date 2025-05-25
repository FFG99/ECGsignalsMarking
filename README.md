# ECGsignalsMarking

API для анализа и классификации состояний по ЭКГ сигналам.

## Описание

Проект представляет собой API для:

1. Загрузки и хранения записей ЭКГ
2. Извлечения признаков из ЭКГ сигналов
3. Классификации состояний (покой, нагрузка, восстановление) по ЭКГ

В проекте уже есть обученная модель, которая готова к использованию. При желании вы можете обучить модель на своих данных, следуя инструкциям ниже.

В директории `notebooks/` вы можете найти Jupyter исследовательские ноутбуки и ноутбуки с тестированием API.

## Требования

- Python 3.8+
- Docker
- Docker Compose
- PostgreSQL 15

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/FFG99/ECGsignalsMarking && cd ECGsignalsMarking
```

2. Создайте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

1. Запустите PostgreSQL в Docker:

```bash
docker-compose up -d
```

2. Примените миграции:

```bash
alembic upgrade head
```

3. Запустите сервер:

```bash
uvicorn app.main:app --reload
```

## Обучение модели

Если вы хотите обучить модель на своих данных:

1. Подготовьте файл разметки `data_markup.json` в формате:

```json
{
    "path/to/file.edf": {
        "rest": [[start_time, end_time], ...],
        "load": [[start_time, end_time], ...],
        "recovery": [[start_time, end_time], ...]
    }
}
```

2. Запустите обучение:

```bash
python train.py
```

Модель будет сохранена в директории `model/` и заменит существующую.

## API Endpoints

### POST /records/upload

Загрузка EDF файла в базу данных.

**Параметры:**

- `file`: EDF файл с записью ЭКГ

**Ответ:**

```json
{
    "id": 1,
    "filename": "example.edf",
    "created_at": "2024-03-14T12:00:00",
    "updated_at": "2024-03-14T12:00:00"
}
```

### GET /records/ {record_id}/data

Получение данных ЭКГ из базы данных по ID записи.

**Ответ:**

```json
{
    "data": [[...], ...]  // Массив значений ЭКГ
}
```

### GET /records/

Получение списка всех записей ЭКГ.

**Ответ:**

```json
[
    {
        "id": 1,
        "filename": "example.edf",
        "created_at": "2024-03-14T12:00:00",
        "updated_at": "2024-03-14T12:00:00"
    }
]
```

### DELETE /records/ {record_id}

Удаление записи ЭКГ из базы данных.

### POST /records/ {record_id}/predict

Предсказание состояний для записи ЭКГ.

**Ответ:**

```json
{
    "record_id": 1,
    "predictions": [
        {
            "start_time": 0.0,
            "end_time": 10.0,
            "state": "rest"
        }
    ]
}
```
