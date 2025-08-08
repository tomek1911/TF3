
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG HOST_UID
ARG HOST_GID

RUN if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then \
      groupadd -g $HOST_GID user && \
      useradd -m --no-log-init -u $HOST_UID -g $HOST_GID user && \
      mkdir -p /opt/app /input /output /output/metadata && \
      chown -R $HOST_UID:$HOST_GID /opt/app /input /output; \
    else \
      groupadd -r user && \
      useradd -m --no-log-init -r -g user user && \
      mkdir -p /opt/app /input /output /output/metadata && \
      chown user:user /opt/app /input /output; \
    fi

USER user

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT ["python", "-m", "process"]