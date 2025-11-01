uvicorn openai_proxy.main:app --reload --port 8080 --http openai_proxy.http_logging:LoggingH11Protocol
