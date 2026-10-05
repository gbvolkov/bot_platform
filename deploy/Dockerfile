# syntax=docker/dockerfile:1.7
FROM python:3.13.3-slim-bookworm AS access-build
WORKDIR /build
COPY deploy/requirements.access.lock deploy/requirements.access.lock
RUN python -m pip install --no-cache-dir -r deploy/requirements.access.lock
COPY platform_contracts platform_contracts
COPY config_defaults config_defaults
COPY platform_sdk platform_sdk
COPY platform_application platform_application
COPY platform_infrastructure platform_infrastructure
COPY platform_client platform_client
COPY platform_access platform_access
COPY openai_proxy openai_proxy
COPY web_chat web_chat
RUN python -m pip install --no-build-isolation --no-deps ./platform_contracts ./platform_sdk ./platform_application ./platform_infrastructure ./platform_client ./platform_access ./openai_proxy ./web_chat

FROM python:3.13.3-slim-bookworm AS access
COPY --from=access-build /usr/local /usr/local
WORKDIR /app
COPY config_defaults config_defaults
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "platform_access.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12.10-slim-bookworm AS ismart-python

FROM python:3.13.3-slim-bookworm AS tokenizer-data
COPY deploy/fetch_nltk_assets.py /fetch_nltk_assets.py
RUN python /fetch_nltk_assets.py /opt/nltk_data

FROM python:3.13.3-slim-bookworm AS privacy-models
# Palimpsest's existing Russian analyzer uses the large model in addition to
# the small model in uv.lock. Match cheetan's installed 3.8.0 resource exactly.
RUN python -m pip download --no-deps --dest /models "https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.8.0/ru_core_news_lg-3.8.0-py3-none-any.whl#sha256=90bd584be86772b647e1b15cdc62e09d3a7182284ee23da95e19c2bf71aa0189"

FROM node:22.23.3-bookworm-slim AS gmail-mcp
WORKDIR /opt/mcp-gmail
# Preserve the previously installed stdio tool and all transitive versions.
COPY deploy/mcp-gmail/package.json deploy/mcp-gmail/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM python:3.13.3-slim-bookworm AS postal-build
# Match the native library already used on cheetan, including its versioned
# default models. Keep native sources and model downloads out of startup.
ARG LIBPOSTAL_REVISION=25099c506612b34b23b1bfe286ca6321fcf06f35
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates autoconf automake libtool build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /build/libpostal
RUN git init && git remote add origin https://github.com/openvenues/libpostal.git && git fetch --depth 1 origin "$LIBPOSTAL_REVISION" && git checkout --detach FETCH_HEAD
RUN ./bootstrap.sh && ./configure --datadir=/usr/local/share --disable-data-download && make -j2 && make install DESTDIR=/opt/libpostal
RUN src/libpostal_data download all /opt/libpostal/usr/local/share/libpostal && install -D -m 644 LICENSE /opt/libpostal/usr/local/share/licenses/libpostal/LICENSE

FROM python:3.13.3-slim-bookworm AS worker
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential pkg-config libcairo2-dev libglib2.0-0 libgl1 libgomp1 libmagic1 poppler-utils ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=postal-build /opt/libpostal/usr/local /usr/local
# Keep iSmart's existing python3.12 subprocess contract alongside the platform's
# Python 3.13 environment. Do not replace the worker's default interpreter.
COPY --from=ismart-python /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=ismart-python /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/libpython3.12.so.1.0
COPY --from=ismart-python /usr/local/lib/python3.12 /usr/local/lib/python3.12
RUN ldconfig && python3.12 --version && python --version
WORKDIR /app
COPY vendor vendor
COPY deploy/requirements.worker.lock deploy/requirements.worker.lock
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked python -m pip install -r deploy/requirements.worker.lock
RUN --mount=from=privacy-models,source=/models,target=/models python -m pip install --no-deps /models/ru_core_news_lg-3.8.0-py3-none-any.whl
COPY --from=gmail-mcp /usr/local/bin/node /usr/local/bin/node
COPY --from=gmail-mcp /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY --from=tokenizer-data /opt/nltk_data /usr/local/share/nltk_data
COPY . .
COPY --from=gmail-mcp /opt/mcp-gmail/node_modules /app/node_modules
COPY --from=gmail-mcp /opt/mcp-gmail/package.json /opt/mcp-gmail/package-lock.json /app/
RUN python -m pip install --no-deps ./platform_contracts ./platform_sdk ./platform_application ./platform_infrastructure ./platform_client ./platform_capabilities ./agent_runtime ./agents ./generators ./packages/gaz_index_builder && mkdir -p /app/logs
ENV PYTHONUNBUFFERED=1 NLTK_DATA=/usr/local/share/nltk_data NPM_CONFIG_OFFLINE=true NPM_CONFIG_YES=true
CMD ["python", "-m", "agent_runtime.worker"]

FROM python:3.13.3-slim-bookworm AS retrieval
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential pkg-config libcairo2-dev libglib2.0-0 libgl1 libgomp1 libmagic1 poppler-utils ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=postal-build /opt/libpostal/usr/local /usr/local
RUN ldconfig
WORKDIR /app
COPY vendor vendor
COPY deploy/requirements.retrieval.lock deploy/requirements.retrieval.lock
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked python -m pip install -r deploy/requirements.retrieval.lock
RUN --mount=from=privacy-models,source=/models,target=/models python -m pip install --no-deps /models/ru_core_news_lg-3.8.0-py3-none-any.whl
COPY --from=tokenizer-data /opt/nltk_data /usr/local/share/nltk_data
COPY platform_contracts platform_contracts
COPY config_defaults config_defaults
COPY platform_sdk platform_sdk
COPY platform_client platform_client
COPY platform_capabilities platform_capabilities
COPY platform_guardrails platform_guardrails
COPY platform_tools platform_tools
COPY platform_utils platform_utils
COPY user_manager user_manager
COPY services/kb_manager services/kb_manager
COPY services/sales_lead_retrieval services/sales_lead_retrieval
COPY packages/gaz_index_builder packages/gaz_index_builder
COPY config.py models.toml ./
COPY data/config data/config
RUN python -m pip install --no-deps ./platform_contracts ./platform_sdk ./platform_client ./platform_capabilities ./packages/gaz_index_builder
RUN mkdir -p /app/logs
ENV PYTHONUNBUFFERED=1 NLTK_DATA=/usr/local/share/nltk_data
CMD ["python", "-m", "uvicorn", "services.sales_lead_retrieval.main:app", "--host", "0.0.0.0", "--port", "8010"]
