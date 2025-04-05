from openai import OpenAI

client = OpenAI(
    base_url = "https://api.scaleway.ai/eee356c0-1e91-41ea-94b2-ae1d58b2b91e/v1",
    api_key = "9d3a3772-239f-4631-88bf-9cd71ee5debb" # Replace SCW_SECRET_KEY with your IAM API key
)

response = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        { "role": "system", "content": "You are a helpful assistant" },
		{ "role": "user", "content": "Tell me something" },
    ],
    max_tokens=512,
    temperature=0,
    top_p=0.95,
    presence_penalty=0,
    stream=True,
)
for chunk in response:
  if chunk.choices and chunk.choices[0].delta.content:
    print(chunk.choices[0].delta.content, end="", flush=True)