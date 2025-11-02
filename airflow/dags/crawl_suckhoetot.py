import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from airflow.operators.python import PythonOperator
from bs4 import BeautifulSoup
from loguru import logger
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from airflow import DAG

sys.path.append(os.path.abspath("/opt"))
from utils import WebdriverFactory

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["you@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def crawl_suckhoetot_news(
    output_csv: str = "",
    browser: str = "",
    driver_path: str = "",
    start_page: int = 1,
    end_page: int = 3,
    **kwargs,
):
    """Crawl health news from SucKhoeTot.vn"""
    # Load existing data if available:
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        logger.info(f"Loaded {len(existing_df)} existing records from {output_csv}")
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
    driver = WebdriverFactory.create_webdriver(
        browser=browser, headless=True, timeout=30, driver_path=driver_path
    )

    try:
        for i in range(start_page, end_page + 1):
            logger.info(f"\nLoading page {i}...")
            try:
                driver.get(f"https://suckhoetot.vn/cat{i}")
            except TimeoutException:
                logger.warning("Page load timed out, skipping waiting")

            # Wait for articles to appear
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "article"))
                )
            except TimeoutException:
                logger.warning("Articles did not appear in time, skipping...")

            soup = BeautifulSoup(driver.page_source, "lxml")
            articles = soup.find_all("article")
            logger.info(f"Found {len(articles)} articles.")

            for art in articles:
                title_tag = art.find("h3", class_="elementor-post__title") or art.find(
                    "h2", class_="elementor-post__title"
                )
                link_tag = title_tag.find("a") if title_tag else None
                summary_tag = art.find("p", class_="description")

                title = link_tag.text.strip() if link_tag else None
                url = link_tag.get("href") if link_tag else None
                summary = summary_tag.text.strip() if summary_tag else None

                # Extract publication time
                time_tag = art.find("span", class_="elementor-post-date")
                published_date = None
                if time_tag:
                    text = time_tag.get_text(strip=True)
                    match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2})", text)
                    if match:
                        published_str = f"{match.group(1)} {match.group(2)}"
                        published_date = datetime.strptime(published_str, "%d/%m/%Y %H:%M")

                if not title or not url:
                    continue

                # Skip duplicate URLs
                if not existing_df.empty and url in existing_df["url"].values:
                    logger.warning(f"Skip this page: {url}")
                    continue

                # Initialize news dictionary
                news = {
                    "title": title,
                    "url": url,
                    "source": "suckhoetot",
                    "summary": summary,
                    "content": None,
                    "published_at": published_date,
                    "crawled_at": datetime.now(timezone.utc),
                    "is_fake": True,
                }
                # print(news)

                # Load article detail page
                try:
                    driver.get(url)
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "html"))
                    )
                except TimeoutException:
                    logger.warning(f"Timeout when loading {url}, skipping this article.")
                    # continue

                detail_soup = BeautifulSoup(driver.page_source, "lxml")

                # Extract content
                content_section = detail_soup.find(
                    "section",
                    class_="elementor-section elementor-top-section elementor-element elementor-element-c5969be elementor-section-boxed elementor-section-height-default elementor-section-height-default",
                )
                if content_section:
                    paragraphs = content_section.find_all("p")
                    content = "\n".join(
                        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                    )
                    news["content"] = content
                else:
                    logger.warning(f"Content not found in {url}")

                # Save to CSV immediately
                new_row = pd.DataFrame([news])
                existing_df = pd.concat([existing_df, new_row]).drop_duplicates(
                    subset=["url"], keep="first"
                )
                existing_df.to_csv(output_csv, index=False, encoding="utf-8")

                logger.info(f"Saved article: {title[:80]}")

                # Avoid rate limiting
                time.sleep(1)

    finally:
        driver.quit()
        logger.info("Driver closed.")


with DAG(
    dag_id="crawl_suckhoetot_news",
    default_args=default_args,
    description="Crawl Suckhoetot.vn Health News and save to CSV",
    schedule="0 6 * * *",  # every day at 6:00 AM
    start_date=datetime(2025, 10, 20),
    catchup=False,
    tags=["crawl", "selenium", "suckhoetot"],
) as dag:

    def crawl_task(**context):
        output_csv = "/opt/data/suckhoetot_news.csv"
        browser = "chrome"
        driver_path = "chromedriver-linux64/chromedriver"

        crawl_suckhoetot_news(
            output_csv=output_csv,
            browser=browser,
            driver_path=driver_path,
            start_page=1,
            end_page=7,
        )

    crawl_suckhoetot = PythonOperator(
        task_id="crawl_suckhoetot_task",
        python_callable=crawl_task,
    )

    crawl_suckhoetot
