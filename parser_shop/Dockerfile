FROM ubuntu:latest
MAINTAINER Yours name <Yours mail>

RUN apt-get update

RUN apt-get install -y tzdata
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV TELEGRAM_API_TOKEN='Yours key'
RUN mkdir /usr/src/ParserShop

WORKDIR /usr/src/ParserShop

RUN apt-get update
RUN apt-get install -y python3 python3-pip && \
    pip3 install poetry
RUN poetry config virtualenvs.create false

COPY parser_shop /usr/src/ParserShop/parser_shop
COPY pyproject.toml poetry.lock /usr/src/ParserShop

RUN poetry install

#USER node
CMD ["poetry", "run", "python3", "parser_shop/script/start.py"]
#RUN adduser --home /usr/src/ParserShop defuser
