FROM node:22.23.3-bookworm-slim AS frontend-build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY . .
ENV VITE_API_BASE_URL=/api
RUN npx vite build --config vite.deploy.config.ts

FROM nginx:1.28.0-alpine AS frontend
COPY nginx.deploy.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /app/dist-web /usr/share/nginx/html

FROM python:3.13.3-slim-bookworm AS backend
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libcairo2-dev libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.deploy.lock /app/requirements.deploy.lock
RUN pip install --no-cache-dir -r requirements.deploy.lock
COPY backend/app app
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8009"]
