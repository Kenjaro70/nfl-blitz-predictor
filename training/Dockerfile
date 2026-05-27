FROM jupyter/base-notebook:python-3.11

USER root
WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER jovyan

EXPOSE 8888
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
