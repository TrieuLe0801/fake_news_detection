import re
import time
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src import HealthNews
from utils import WebdriverSingleton

driver_singleton = WebdriverSingleton(
    browser="chrome", headless=True, timeout=20, driver_path="tests/chromedriver"
)
driver = driver_singleton.get_driver()

driver.set_page_load_timeout(20)
try:
    driver.get("https://vnexpress.net/suc-khoe")
except TimeoutException:
    print("Page load timed out, continuing...")

# Wait until recognizing article
WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.TAG_NAME, "article")))

html = driver.page_source

soup = BeautifulSoup(html, "lxml")
articles = soup.find_all("article")

print(f"Found {len(articles)} articles")

news_items = []

for art in articles:
    title_tag = art.find("h3", class_="title-news") or art.find("h2", class_="title-news")
    link_tag = title_tag.find("a") if title_tag else None
    summary_tag = art.find("p", class_="description")

    title = link_tag.text.strip() if link_tag else None
    url = link_tag.get("href") if link_tag else None
    summary = summary_tag.text.strip() if summary_tag else None

    if not title or not url:
        continue

    # Create Dictionary 
    news = {
        "title": title,
        "url": url,
        "source": "VnExpress",
        "summary": summary,
        "publisummary": None,  
        "summary": None, 
        "crawled_at": datetime.now(timezone.utc),
    }

    print(news["url"])
    try:
        driver.get(news["url"])
    except TimeoutException:
        print(f"Page {news['url']} load timed out, continuing...")

    WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

    detail_soup = BeautifulSoup(driver.page_source, "lxml")

    # Get content
    content_section = detail_soup.find("section", class_="section page-detail top-detail")
    if content_section:
        paragraphs = content_section.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        news["content"] = content
    else:
        print(f" Cannot find out section.page-detail.top-detail in {news['url']}")
        news["content"] = None

    # Get time
    time_tag = detail_soup.find("span", class_="date")
    if time_tag:
        text = time_tag.get_text(strip=True)
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2})", text)
        if date_match:
            published_str = f"{date_match.group(1)} {date_match.group(2)}"
            news["published_at"] = datetime.strptime(published_str, "%d/%m/%Y %H:%M")

    time.sleep(1)  # Ignore IP restriction

    # Create HealthNews and initalize values
    valid_keys = {c.name for c in HealthNews.__table__.columns}
    filtered_news = {k: v for k, v in news.items() if k in valid_keys}
    healthnews = HealthNews(**filtered_news)

    print(healthnews.__repr__())
    print(healthnews.summary)
    print(healthnews.url)
    print(healthnews.crawled_at)
    print(healthnews.content)
    print(healthnews.published_at)
    print("-" * 18)
    news_items.append(healthnews)

driver.quit()
