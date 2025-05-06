# ECGsignalsMarking API

## Требования

- Python 3.8+
- Docker
- Docker Compose

## Установка

1. Клонируйте репозиторий

```bash
git clone https://github.com/FFG99/ECGsignalsMarking && cd ECGsignalsMarking
```

1. Создайте виртуальное окружение:

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

## API Endpoints

### POST /upload/

Загрузка EDF файла в базу данных.

### GET /records/ {record_id}/data

Получение данных EEG из базы данных по ID записи.

### GET /records/

Получение данных о всех ЭКГ в базы данных.
