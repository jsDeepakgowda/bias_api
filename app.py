from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API Key not found! Check your .env file.")
# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to call the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API setup
openai.api_key = OPENAI_API_KEY

# Define request model
class ArticleRequest(BaseModel):
    headline: str
    paragraphs: list[str]

@app.post("/analyze")
async def analyze_article(article: ArticleRequest):
    article_text = f"Title: {article.headline}\n\n" + "\n\n".join(article.paragraphs)

    prompt = f"""
    You are BiasBuster, an AI that analyzes news articles for bias and rewrites them to be neutral.

    **Instructions:**
    - **Detect bias:** Identify loaded language, emotional framing, misleading statements, missing context, and factual inconsistencies.
    - **Categorize bias:** Label each biased word/phrase with a category (e.g., Political, Emotional, Sensational, Misleading).
    - **Score the article:** Provide a bias score (0-10) and political leaning (left, right, neutral).
    - **Rewrite neutrally:** Provide an unbiased, fact-checked version while preserving original intent and readability.

    **Return ONLY valid JSON without any extra text or formatting.**  

    **JSON FORMAT:**  
    {{
      "analysis": {{
        "biasScore": number,
        "politicalLeaning": string,
        "biasedLanguage": [array of biased words/phrases],
        "biasCategories": [array of categories matching biasedLanguage],
        "biasExplanations": [array of explanations matching biasedLanguage],
        "missingContext": [array of missing contexts],
        "factualIssues": [array of factual issues]
      }},
      "rewrittenArticle": {{
        "headline": string,
        "content": [array of rewritten paragraphs]
      }},
      "sources": [array of source objects with title and url]
    }}

    **Here is the article:**  
    {article_text}
    """.strip()

    try:
        response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",  

            messages=[
                {"role": "system", "content": "You are a bias analysis AI that returns JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )

        result = response
        return result  # Send JSON response back to frontend

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
