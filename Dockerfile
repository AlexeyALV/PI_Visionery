FROM python:3.9

# This flag is important to output python logs correctly in docker!
ENV PYTHONUNBUFFERED 1
# Flag to optimize container size a bit by removing runtime python cache
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /visionery

COPY ./requirements.txt /visionery/requirements.txt
RUN pip install -r /visionery/requirements.txt
COPY ./main.py /visionery/main.py
COPY ./templates/index.html /visionery/templates/index.html
COPY ./static/empty.txt /visionery/static/empty.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["main.py"]