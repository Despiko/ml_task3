FROM python:3-slim

WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow
EXPOSE 5050

#ENV BACKEND_URI sqlite:////mlflow/mlflow.db
#ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --host 0.0.0.0 --port 5050
#--default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5050
#--backend-store-uri ${BACKEND_URI}