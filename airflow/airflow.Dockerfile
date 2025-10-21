ARG AIRFLOW_IMAGE_NAME
FROM ${AIRFLOW_IMAGE_NAME}
# FROM apache/airflow:3.1.0-python3.13

ARG AIRFLOW_USER_HOME=/usr/local/airflow
ENV PYTHONPATH=$PYTHONPATH:${AIRFLOW_USER_HOME}

WORKDIR ${AIRFLOW_USER_HOME}

USER root
RUN groupadd --gid 999 docker \
   && usermod -aG docker airflow

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  chromium \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY airflow.requirements.txt .

USER airflow

RUN pip install --no-cache-dir -r airflow.requirements.txt





