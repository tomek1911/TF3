
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

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

ENV PYDEVD_DISABLE_FILE_VALIDATION=1
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
# RUN python -m pip install --user debugpy

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user checkpoints/ /opt/app/checkpoints/

ENTRYPOINT ["python", "-m", "process"]
# ENTRYPOINT ["python", "-Xfrozen_modules=off", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "process"]