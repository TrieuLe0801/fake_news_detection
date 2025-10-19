import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src import HealthNews
from utils import WebdriverSingleton


def crawl_vnexpress_health(output_csv="tests/vnxpress_news.csv", start_page=1, end_page=3):
    """Crawl health news from VnExpress."""

    # Load existing data if available
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        print(f"Loaded {len(existing_df)} existing records from {output_csv}")
    else:
        existing_df = pd.DataFrame(
            columns=[
                "title",
                "url",
                "source",
                "summary",
                "content",
                "published_at",
                "crawled_at",
                "is_fake",
            ]
        )

    # Initialize WebDriver (Singleton)
    driver_singleton = WebdriverSingleton(
        browser="chrome", headless=True, timeout=30, driver_path="tests/chromedriver"
    )
    driver = driver_singleton.get_driver()

    try:
        for i in range(start_page, end_page + 1):
            print(f"\nLoading page {i}...")
            try:
                driver.get(f"https://vnexpress.net/suc-khoe-p{i}")
            except TimeoutException:
                print("Page load timed out, skipping this page.")

            # Wait for articles to appear
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "article"))
                )
            except TimeoutException:
                print("No articles found on this page, skipping.")

            soup = BeautifulSoup(driver.page_source, "lxml")
            articles = soup.find_all("article")
            print(f"Found {len(articles)} articles.")

            for art in articles:
                title_tag = art.find("h3", class_="title-news") or art.find(
                    "h2", class_="title-news"
                )
                link_tag = title_tag.find("a") if title_tag else None
                summary_tag = art.find("p", class_="description")

                title = link_tag.text.strip() if link_tag else None
                url = link_tag.get("href") if link_tag else None
                summary = summary_tag.text.strip() if summary_tag else None

                if not title or not url:
                    continue

                # Skip duplicate URLs
                if not existing_df.empty and url in existing_df["url"].values:
                    print(f"Skip this page: {url}")
                    continue

                # Initialize news dictionary
                news = {
                    "title": title,
                    "url": url,
                    "source": "VnExpress",
                    "summary": summary,
                    "content": None,
                    "published_at": None,
                    "crawled_at": datetime.now(timezone.utc),
                    "is_fake": False,
                }

                # Load article detail page
                try:
                    driver.get(url)
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "html"))
                    )
                except TimeoutException:
                    print(f"Timeout when loading {url}, skipping this article.")
                    # continue

                detail_soup = BeautifulSoup(driver.page_source, "lxml")

                # Extract content
                content_section = detail_soup.find(
                    "section", class_="section page-detail top-detail"
                )
                if content_section:
                    paragraphs = content_section.find_all("p")
                    content = "\n".join(
                        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                    )
                    news["content"] = content
                else:
                    print(f"Content not found in {url}")

                # Extract publication time
                time_tag = detail_soup.find("span", class_="date")
                if time_tag:
                    text = time_tag.get_text(strip=True)
                    match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2})", text)
                    if match:
                        published_str = f"{match.group(1)} {match.group(2)}"
                        news["published_at"] = datetime.strptime(published_str, "%d/%m/%Y %H:%M")

                # Save to CSV immediately
                new_row = pd.DataFrame([news])
                existing_df = pd.concat([existing_df, new_row]).drop_duplicates(
                    subset=["url"], keep="first"
                )
                existing_df.to_csv(output_csv, index=False, encoding="utf-8")

                print(f"Saved article: {title[:80]}")

                # Avoid rate limiting
                time.sleep(1)

    finally:
        driver.quit()
        print("Driver closed.")


if __name__ == "__main__":
    crawl_vnexpress_health(start_page=1, end_page=20)
