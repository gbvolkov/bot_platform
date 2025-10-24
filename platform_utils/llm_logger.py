# logging_callback.py
from langchain_core.callbacks import BaseCallbackHandler
import json, time

class JSONFileTracer(BaseCallbackHandler):
    def __init__(self, path="traces.jsonl"):
        self.f = open(path, "a", encoding="utf-8")
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.f.write(json.dumps({"ts": time.time(), "type":"llm_start",
                                 "model": serialized, "prompt": prompts}, ensure_ascii=False) + "\n")
    def on_llm_end(self, response, **kwargs):
        # response.generations contains the LLM outputs
        self.f.write(json.dumps({"ts": time.time(), "type":"llm_end",
                                 "response": [g.text for g in response.generations[0]]}, ensure_ascii=False) + "\n")
        self.f.flush()
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.f.write(json.dumps({"ts": time.time(), "type":"tool_start",
                                 "tool": serialized["name"], "input": input_str}, ensure_ascii=False) + "\n")
    def on_tool_end(self, output, **kwargs):
        try:
            # ToolMessage has .content and .tool_call_id; adjust as needed
            output_dict = {"content": getattr(output, "content", str(output)),
                        "tool_call_id": getattr(output, "tool_call_id", None)}
            self.f.write(json.dumps({"ts": time.time(),
                                    "type":"tool_end",
                                    "output": output_dict}, ensure_ascii=False) + "\n")
        except Exception as err:
            # Fallback: write a simplified string if serialization fails
            self.f.write(json.dumps({"ts": time.time(),
                                    "type":"tool_end",
                                    "output": str(output),
                                    "error": str(err)}, ensure_ascii=False) + "\n")
        self.f.flush()
