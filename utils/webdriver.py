import atexit

from fake_useragent import UserAgent
from selenium import webdriver

# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class WebdriverSingleton:
    _instance = None
    _driver = None

    def __new__(
        cls,
        browser: str = "chrome",
        headless: bool = False,
        timeout: int = 20,
        driver_path: str = "",
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._driver = cls._init_driver(browser, headless, timeout, driver_path)
            atexit.register(cls._close_driver)  # Close all driver when finish
        return cls._instance

    @classmethod
    def _init_driver(cls, browser, headlless, timeout, driver_path):
        user_agent = UserAgent().random
        options = None
        service = None

        if browser == "chrome":
            options = webdriver.ChromeOptions()
            service = webdriver.ChromeService(executable_path=driver_path)
        elif browser == "edge":
            options = webdriver.EdgeOptions()
            service = webdriver.EdgeService(executable_path=driver_path)
        elif browser == "firefox":
            options = webdriver.FirefoxOptions()
            service = webdriver.FirefoxService(executable_path=driver_path)
        else:
            raise ValueError("Unsupported browser. Use 'chrome', 'edge', or 'firefox'.")

        if headlless:
            options.add_argument("--headless=new")

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--incognito")
        options.add_argument("--disable-gpu")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"user-agent={user_agent}")

        if browser == "chrome":
            driver = webdriver.Chrome(service=service, options=options)
        if browser == "edge":
            driver = webdriver.Edge(service=service, options=options)
        else:
            driver = webdriver.Firefox(service=service, options=options)

        driver.set_page_load_timeout(timeout)
        driver.maximize_window()

        return driver

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            raise RuntimeError("WebDriver not initialized yet. Create an instance first.")
        return cls._driver

    @classmethod
    def _close_driver(cls):
        if cls._driver:
            try:
                cls._driver.quit()
                print("WebDriver closed.")
            except Exception:
                pass
            cls._driver = None
