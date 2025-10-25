import atexit

from fake_useragent import UserAgent
from selenium import webdriver


class WebdriverFactory:
    @staticmethod
    def create_webdriver(
        browser: str = "chrome", headless: bool = True, timeout: int = 30, driver_path: str = ""
    ):
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

        if headless:
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
