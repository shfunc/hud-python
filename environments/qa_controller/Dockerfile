FROM python:3.10-alpine

# Install build dependencies
RUN apk add --no-cache build-base

# Create directories
RUN mkdir -p /tmp/state /environment

# Set working directory
WORKDIR /environment

# install controller
COPY . /controller
RUN pip install -e /controller --break-system-packages

# note: when you make changes to the controller, hud-sdk will automatically reinstall the python package corresponding to the controller before every step.
# Any changes to the Dockerfile though will only be reflected on the next environment creation

# Keep container running
CMD ["tail", "-f", "/dev/null"]
