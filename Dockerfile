FROM ghcr.io/prefix-dev/pixi:0.57.0 AS build

RUN apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y libssl-dev

WORKDIR /app
COPY . .
RUN pixi install --locked -e default
RUN pixi shell-hook -e default -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh
RUN chmod 0755 /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["python"]
