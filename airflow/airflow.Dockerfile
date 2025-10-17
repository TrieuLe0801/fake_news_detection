ARG AIRFLOW_IMAGE_NAME
FROM ${AIRFLOW_IMAGE_NAME}

ARG AIRFLOW_USER_HOME=/usr/local/airflow
ENV PYTHONPATH=$PYTHONPATH:${AIRFLOW_USER_HOME}

WORKDIR ${AIRFLOW_USER_HOME}

USER root
RUN groupadd --gid 999 docker \
   && usermod -aG docker airflow

COPY airflow.requirements.txt .

RUN pip install --no-cache-dir -r airflow.requirements.txt

USER airflow





