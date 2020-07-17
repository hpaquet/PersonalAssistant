FROM python:3.7.4

COPY . /alice

WORKDIR /alice

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python setup.py install

CMD [ "python", "./alice/run_container.py" ]