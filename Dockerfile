FROM python:3.10-alpine

WORKDIR /app

COPY . /app
COPY requirements.txt /app/

RUN apk add --no-cache gcc musl-dev linux-headers

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
