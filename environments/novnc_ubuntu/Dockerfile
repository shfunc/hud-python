FROM govindhud/novnc-ubuntu:latest
COPY . /controller
RUN pip install -e /controller --break-system-packages

# note: when you make changes to the controller, hud-sdk will automatically reinstall the python package corresponding to the controller before every step.
# Any changes to the Dockerfile though will only be reflected on the next environment creation