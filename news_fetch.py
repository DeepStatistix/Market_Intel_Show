# file: news_fetch.py
import requests
import logging

def fetch_agri_news():
    """
    Fetch the latest agriculture-related news from an API (e.g. NewsAPI).
    Replace <YOUR_NEWSAPI_KEY> with your actual key or store it in environment variables.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "agriculture OR farming OR agri",
        "apiKey": "<YOUR_NEWSAPI_KEY>",
        "language": "en",
        "pageSize": 5  # limit number of articles
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {str(e)}")
        return []
