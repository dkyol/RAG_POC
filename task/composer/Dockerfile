# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y build-essential bash curl unzip wget git libgl1-mesa-glx libglib2.0-0 

#Run pip dependencies
COPY composer/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install ipywidgets jupyterlab==4.0.0 jupyter-archive==3.4.0

ADD llm_client/ai-endpoints ai-endpoints
RUN cd ai-endpoints && pip3 install -e .

# COPY notebooks
WORKDIR /dli/task
# COPY . .

# Expose port 8888 for JupyterLab
EXPOSE 8888 
EXPOSE 9012

# Start JupyterLab when the container runs
# --ip 0.0.0.0: Run on localhost
# --allow-root: Enable the use of sudo commands in the notebook
# --no-browser: Do not launch a browser by default
# --NotebookApp.base_url: Allow value to be passed in for production
# --NotebookApp.token: Do not require token to access notebook
# --NotebookApp.password: Do not require password to run jupyter server
ENTRYPOINT ["jupyter", "lab", \
    "--ip", "0.0.0.0", \
    "--allow-root", \
    "--no-browser", \
    "--NotebookApp.base_url=/lab", \
    "--NotebookApp.token=''", \
    "--NotebookApp.password=''"]