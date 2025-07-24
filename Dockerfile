FROM ultralytics/ultralytics:latest

WORKDIR /project 

COPY . .

RUN python3 -m pip install -r requirements.txt

CMD ["python3", "./main.py"]

EXPOSE 8009
