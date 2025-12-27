from openai import OpenAI
import os

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY first")

client = OpenAI(api_key=api_key)
ids = [m.id for m in client.models.list().data]
print("\n".join(ids))
