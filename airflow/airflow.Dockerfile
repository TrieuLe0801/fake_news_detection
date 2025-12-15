# ARG AIRFLOW_IMAGE_NAME
# FROM ${AIRFLOW_IMAGE_NAME}
FROM apache/airflow:3.1.0

ARG AIRFLOW_USER_HOME=/opt/airflow

ENV PYTHONPATH=$PYTHONPATH:${AIRFLOW_USER_HOME}:/usr/local/bin

WORKDIR ${AIRFLOW_USER_HOME}

USER root
# Check if group with GID 999 exists, if not create new one.
RUN if ! getent group 999 >/dev/null; then \
        groupadd --gid 999 docker; \
    fi \
    && usermod -aG docker airflow

RUN apt-get update && apt-get install -y \
    unzip \
    tar \
    wget \
    gnupg \
    libfuse2 \
    default-jdk \
    --no-install-recommends

# Install Chrome and ChromeDriver
RUN wget -O /opt/google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y /opt/google-chrome-stable_current_amd64.deb && rm /opt/google-chrome-stable_current_amd64.deb

RUN wget -O /opt/chromedriver-linux64.zip https://storage.googleapis.com/chrome-for-testing-public/143.0.7499.42/linux64/chromedriver-linux64.zip \
    && unzip -o /opt/chromedriver-linux64.zip && rm /opt/chromedriver-linux64.zip

# Install Edge and EdgeDriver
RUN wget -O - https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
    install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ && \
    sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge.list' && \
    rm microsoft.gpg

RUN apt-get update && apt-get install -y microsoft-edge-stable

RUN wget -O /opt/edgedriver-linux64.zip https://msedgedriver.microsoft.com/141.0.3537.99/edgedriver_linux64.zip \
    && unzip -o /opt/edgedriver-linux64.zip && rm /opt/edgedriver-linux64.zip
  
# # Install Firefox and GeckoDriver
# RUN apt install -d -m 0755 /etc/apt/keyrings

RUN wget -O /opt/geckodriver-linux.tar.gz https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz \
    && tar -xvf /opt/geckodriver-linux.tar.gz --overwrite && chmod +x geckodriver && rm /opt/geckodriver-linux.tar.gz

# RUN apt-get install -y --no-install-recommends \

RUN apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY airflow.requirements.txt .

USER airflow

RUN pip install --no-cache-dir -r airflow.requirements.txt





