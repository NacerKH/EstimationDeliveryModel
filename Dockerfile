FROM python:3.8
WORKDIR /app
COPY . /app

VOLUME /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD ["tail", "-f", "/dev/null"]

