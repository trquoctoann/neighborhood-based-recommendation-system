FROM python:3.8.5-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 5000
CMD ["python3", "Recommendation System.py"]