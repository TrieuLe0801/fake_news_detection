import io
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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airflow import DAG

sys.path.append(os.path.abspath("/opt"))
from dotenv import load_dotenv

from src import Base, HealthNews
from utils import WebdriverFactory

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["you@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def insert_or_update(df: pd.DataFrame, engine: object = engine, mode: str = "upsert"):
    """
    Bulk insert a pandas DataFrame into PostgreSQL using COPY with ON CONFLICT handling.

    Args:
        df (pd.DataFrame): Data to insert.
        engine (object): SQLAlchemy engine connected to the PostgreSQL database.
        mode (str): "upsert" (insert or update existing) or "ignore" (skip duplicates).
    """
    # Ensure only valid columns from model are included
    # Ensure table exists (create it if missing)
    Base.metadata.create_all(engine)
    valid_columns = [
        col.name
        for col in HealthNews.__table__.columns
        if not col.primary_key and col.name in df.columns
    ]
    df = df[valid_columns]

    table_name = HealthNews.__tablename__

    with sessionmaker(bind=engine)() as session:
        raw_conn = session.connection().connection
        with raw_conn.cursor() as cursor:
            # Step 1: Create temporary table
            cursor.execute(
                f"CREATE TEMP TABLE temp_{table_name} AS SELECT * FROM {table_name} LIMIT 0;"
            )

            # Step 2: COPY into temporary table
            with io.StringIO() as buffer:
                df.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                cursor.copy_expert(
                    f"""
                    COPY temp_{table_name} ({','.join(df.columns)})
                    FROM STDIN WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '"')
                    """,
                    buffer,
                )

            # Step 3: Merge data with conflict handling
            if mode == "ignore":
                conflict_action = "DO NOTHING"
            elif mode == "upsert":
                update_clause = ", ".join(
                    [f"{col}=EXCLUDED.{col}" for col in df.columns if col != "url"]
                )
                conflict_action = f"DO UPDATE SET {update_clause}"
            else:
                raise ValueError("Invalid mode. Use 'upsert' or 'ignore'.")

            cursor.execute(
                f"""
                INSERT INTO {table_name} ({','.join(df.columns)})
                SELECT {','.join(df.columns)} FROM temp_{table_name}
                ON CONFLICT (url) {conflict_action};
            """
            )

        raw_conn.commit()
    logger.info("✅ Inserted or updated rows successfully.")


def crawl_vnexpress_health(
    output_csv: str = "",
    browser: str = "chrome",
    driver_path: str = "",
    start_page: int = 1,
    end_page: int = 3,
    **kwargs,
):
    """Crawl health news from VnExpress."""

    # Load existing data if available
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
                driver.get(f"https://vnexpress.net/suc-khoe-p{i}")
            except TimeoutException:
                logger.warning("Page load timed out, skipping this page.")

            # Wait for articles to appear
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "article"))
                )
            except TimeoutException:
                logger.warning("No articles found on this page, skipping.")

            soup = BeautifulSoup(driver.page_source, "lxml")
            articles = soup.find_all("article")
            logger.info(f"Found {len(articles)} articles.")

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
                    logger.info(f"Skip this page: {url}")
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
                    logger.warning(f"Timeout when loading {url}, skipping this article.")
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
                    logger.info(f"Content not found in {url}")

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

                logger.info(f"Saved article: {title[:80]}")

                # Avoid rate limiting
                time.sleep(1)

    finally:
        driver.quit()
        logger.info("Driver closed.")

    return existing_df


# ---- Define the DAG ----
with DAG(
    dag_id="vnexpress_health_crawl_dag",
    default_args=default_args,
    description="Crawl VnExpress Health News and save to CSV",
    schedule="0 6 * * *",  # every day at 6:00 AM
    start_date=datetime(2025, 10, 20),
    catchup=False,
    tags=["crawl", "selenium", "vnexpress"],
) as dag:

    def crawl_task(**context):
        output_csv = "/opt/data/vnxpress_news.csv"
        browser = "chrome"
        driver_path = "chromedriver-linux64/chromedriver"

        data_df = crawl_vnexpress_health(
            output_csv=output_csv,
            browser=browser,
            driver_path=driver_path,
            start_page=1,
            end_page=20,
        )
        context["ti"].xcom_push(key="data", value=data_df)

    def insert_and_update_task(**context):
        ti = context["ti"]
        data_df = ti.xcom_pull(task_ids="crawl_vnexpress_health", key="data")
        insert_or_update(data_df, engine=engine, mode="upsert")

    # ---- PythonOperator ----
    crawl_vnexpress = PythonOperator(
        task_id="crawl_vnexpress_health",
        python_callable=crawl_task,
    )

    insert_and_update = PythonOperator(
        task_id="insert_and_update_vnexpress_data",
        python_callable=insert_and_update_task,
    )

    crawl_vnexpress >> insert_and_update
