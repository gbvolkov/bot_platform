FROM python:3.13.3-slim-bookworm AS analysis
# Already declared/pinned monorepo libraries; this image contains no platform config or credentials.
RUN python -m pip install --no-cache-dir pandas==2.3.3 numpy==2.3.4 duckdb==1.5.2 openpyxl==3.1.5 pypdf==6.11.0 et-xmlfile==2.0.0 python-dateutil==2.9.0.post0 pytz==2026.2 tzdata==2026.2 six==1.17.0
USER 65534:65534
WORKDIR /output
