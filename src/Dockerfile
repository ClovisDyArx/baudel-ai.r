FROM python:3.9

ARG host=0.0.0.0
ARG port=1234

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY finetuned_models /app/finetuned_models/

EXPOSE ${port}

CMD ["uvicorn", "main:app", "--host", "${host}", "--port", "${port}"]
