# ARG AIRFLOW_IMAGE_NAME
# FROM ${AIRFLOW_IMAGE_NAME}
FROM apache/airflow:3.1.0

ARG AIRFLOW_USER_HOME=/opt/airflow

ENV PYTHONPATH=$PYTHONPATH:${AIRFLOW_USER_HOME}:/usr/local/bin

WORKDIR ${AIRFLOW_USER_HOME}

USER root
RUN groupadd --gid 999 docker \
   && usermod -aG docker airflow
# RUN usermod -aG docker airflow

RUN apt-get update && apt-get install -y wget unzip
RUN wget -O /opt/google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y /opt/google-chrome-stable_current_amd64.deb && rm /opt/google-chrome-stable_current_amd64.deb

RUN wget -O /opt/chromedriver-linux64.zip https://storage.googleapis.com/chrome-for-testing-public/141.0.7390.122/linux64/chromedriver-linux64.zip \
    && unzip -o /opt/chromedriver-linux64.zip && rm /opt/chromedriver-linux64.zip

# RUN apt-get install -y --no-install-recommends \

RUN apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY airflow.requirements.txt .

USER airflow

RUN pip install --no-cache-dir -r airflow.requirements.txt





