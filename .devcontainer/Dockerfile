FROM python:3.8-slim-buster

RUN apt-get update &&\
    apt-get install -y git 

RUN pip3 install numpy matplotlib tabulate scipy scikit-image sklearn

RUN pip3 install yapf

# For ex4
RUN pip3 install pandas pillow

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install tqdm

RUN pip3 install onnxruntime