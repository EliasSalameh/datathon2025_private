import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
  api_key= os.environ.get("OPEN_AI_KEY")
)

class ResearchPaperExtraction(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure."},
        {"role": "user", "content": "..."}
    ],
    response_format=ResearchPaperExtraction,
)

research_paper = completion.choices[0].message.parsed
print(research_paper)