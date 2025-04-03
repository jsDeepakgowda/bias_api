from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import httpx
import asyncio
import os
import json
from typing import List
import json
import re
import asyncio
from datetime import datetime
from typing import List, Optional
from typing import Union, Dict, Any
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize


# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY= os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

openai.api_key = OPENAI_API_KEY
nltk.download('punkt')


# Stopwords list for keyword extraction
STOPWORDS = {"the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "as", "at", "by", "an", "be"}

def extract_keywords(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in STOPWORDS]
    return " ".join([word for word, _ in Counter(words).most_common(5)])
# Request models
class ArticleRequest(BaseModel):
    headline: str
    paragraphs: List[str]
    url: Optional[str] = None
    website: Optional[str] = None
    logo: Optional[str] = None
    published_date: Optional[str] = None

class SimilarArticlesRequest(BaseModel):
    headline: str
class AnalyzeBatchRequest(BaseModel):
    articles: List[ArticleRequest]
@app.get("/")
async def root():
    return {"message": "BiasBuster API is running!"}

@app.post("/find-similar-and-analyze")
async def find_similar_and_analyze(request: SimilarArticlesRequest):
    query = extract_keywords(request.headline)

    newsapi_url = "https://newsapi.org/v2/everything"
    gnews_url = "https://gnews.io/api/v4/search"

    newsapi_params = {
        "q": query,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 7,
       "apiKey": NEWS_API_KEY
    }

    gnews_params = {
        "q": query,
        "lang": "en",
        "max": 7,
        "token": GNEWS_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            # Fetch articles from NewsAPI
            newsapi_response = await client.get(newsapi_url, params=newsapi_params)
            newsapi_data = newsapi_response.json() if newsapi_response.status_code == 200 else {}

            # Fetch articles from GNews
            gnews_response = await client.get(gnews_url, params=gnews_params)
            gnews_data = gnews_response.json() if gnews_response.status_code == 200 else {}

            articles = []
            if newsapi_data.get("articles"):
                articles.extend(newsapi_data["articles"])
            if gnews_data.get("articles"):
                articles.extend(gnews_data["articles"])

            if not articles:
                raise HTTPException(status_code=404, detail="No similar articles found")

            extracted_articles = []
            for article in articles:
                headline = article.get("title", "No Title")
                paragraphs = [article.get("description", ""), article.get("content", "")]
                article_url = article.get("url", "No URL")
                source_name = article.get("source", {}).get("name", "Unknown Source")
                logo_url = f"https://logo.clearbit.com/{article_url.split('/')[2]}" if article_url != "No URL" else None
                published_at = article.get("publishedAt", "") or article.get("publishedAt")

                if published_at:
                    try:
                        published_at = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        published_at = "Unknown Date"

                extracted_articles.append(ArticleRequest(
                    headline=headline,
                    paragraphs=paragraphs,
                    url=article_url,
                    website=source_name,
                    logo=logo_url,
                    published_date=published_at  
                ))

            analysis_results = await analyze_multiple_articles(extracted_articles)
            return {"articles": analysis_results["results"]}

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="API error")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a bias analysis AI that returns JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2500
        )
        #return response.choices[0].message.content
        return response
# Parse the returned content to ensure it matches the expected format
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def extract_json_from_response(response: Union[str, Dict]) -> Dict[str, Any]:
    """
    Final production-ready JSON extractor with enhanced debugging.
    Handles all OpenAI API response formats and edge cases.
    """
    DEFAULT_RESPONSE = {
        "analysis": {
            "biasScore": 0,
            "politicalLeaning": "neutral",
            "biasedLanguage": [],
            "biasCategories": [],
            "biasExplanations": [],
            "missingContext": [],
            "factualIssues": []
        },
        "rewrittenArticle": {
            "headline": "No headline available",
            "content": ["No content available"]
        },
        "sources": []
    }

    if not response:
        return DEFAULT_RESPONSE

    try:
        # Step 1: Extract the content string
        content = ""
        if isinstance(response, dict):
            if 'choices' in response:
                content = response['choices'][0]['message']['content']
            elif 'content' in response:
                content = response['content']
            else:
                return DEFAULT_RESPONSE
        else:
            content = str(response)

        # Step 2: Extract JSON from markdown code blocks
        json_str = ""
        json_match = re.search(r'```(?:json)?\n?(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            json_str = json_match.group(0) if json_match else content

        # Step 3: Clean the JSON string
        if not json_str:
            return DEFAULT_RESPONSE

        json_str = json_str.strip()
        json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas

        # Step 4: Parse with multiple fallback attempts
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            try:
                parsed = json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                inner_json = re.search(r'\{.*\}', json_str, re.DOTALL)
                if inner_json:
                    try:
                        parsed = json.loads(inner_json.group(0), strict=False)
                    except json.JSONDecodeError as e:
                        return DEFAULT_RESPONSE
                else:
                    return DEFAULT_RESPONSE

        # Validate the parsed structure
        if not isinstance(parsed, dict):
            return DEFAULT_RESPONSE
        
        if 'analysis' not in parsed or 'rewrittenArticle' not in parsed:
            return DEFAULT_RESPONSE

        return parsed

    except Exception as e:
        return DEFAULT_RESPONSE
# Function to clean the response content
@app.post("/pokemon")
async def analyze_multiple_articles(articles: list[ArticleRequest]):
    """
    Analyze multiple articles concurrently.
    """

    async def analyze_single(article: ArticleRequest):
        try:
            raw_response = await analyze_article(article)
            
            result = extract_json_from_response(raw_response)

            # Safely extract all fields with defaults
            analysis = result.get("analysis", {})
            rewritten = result.get("rewrittenArticle", {})
            
            return {
                "biasScore": analysis.get("biasScore", 0),
                "politicalLeaning": analysis.get("politicalLeaning", "neutral"),
                "biasedLanguage": analysis.get("biasedLanguage", []),
                "biasCategories": analysis.get("biasCategories", []),
                "biasExplanations": analysis.get("biasExplanations", []),
                "missingContext": analysis.get("missingContext", []),
                "factualIssues": analysis.get("factualIssues", []),
                "rewrittenHeadline": article.headline,
                "rewrittenContent": article.paragraphs,
                "url": article.url,
                "website": article.website,
                "logo": article.logo,
                "publishedAt":article.published_date,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "url": article.url,
                "website": article.website,
                "logo": article.logo
            }

    try:
        results = await asyncio.gather(*[analyze_single(article) for article in articles])
        return {"results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
