uvicorn openai_proxy.main:app --no-reload --port 8084 --http openai_proxy.http_logging:LoggingH11Protocol
