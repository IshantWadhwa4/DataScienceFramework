FROM python:3.7
WORKDIR /DataScienceFrameWork

COPY requirment.txt ./requirment.txt
RUN pip3 install -r requirment.txt

EXPOSE 8501

COPY . /DataScienceFrameWork

ENTRYPOINT [ "streamlit","run" ]
CMD ["run.py"]





