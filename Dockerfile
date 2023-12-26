FROM python:3.11
WORKDIR /code
COPY ./ /code/
RUN pip install --no-cache-dir -r /code/requirements.txt
CMD ["python3", "main.py"]
