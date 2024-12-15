# BACKUP CODE in Selenium

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException  # Importing the TimeoutException

# def fetch_real_time_price(stock_code):
#     # Setup WebDriver
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--ignore-ssl-errors=yes')
#     options.add_argument('--ignore-certificate-errors')
#     driver = webdriver.Chrome(service=service, options=options)
    
#     try:
#         driver.get(f'https://finance.yahoo.com/quote/{stock_code}?guccounter=1')
#         wait = WebDriverWait(driver, 60)

#          # XPath locators
#         price_locator = (By.XPATH, '//fin-streamer[@data-field="regularMarketPrice"]')
#         change_locator = (By.XPATH, '//fin-streamer[@data-field="regularMarketChange"]')
#         percentage_locator = (By.XPATH, '//fin-streamer[@data-field="regularMarketChangePercent"]')

#         # Fetching data with exception handling for timeouts
#         try:
#             price = wait.until(EC.visibility_of_element_located(price_locator)).text
#         except TimeoutException:
#             print("Failed to locate the price element within the given time.")
#             price = "N/A"  # Fallback value if the timeout exception is raised
        
#         try:
#             change = wait.until(EC.visibility_of_element_located(change_locator)).text
#         except TimeoutException:
#             print("Failed to locate the change element within the given time.")
#             change = "N/A"
        
#         try:
#             percentage = wait.until(EC.visibility_of_element_located(percentage_locator)).text
#         except TimeoutException:
#             print("Failed to locate the percentage change element within the given time.")
#             percentage = "N/A"

#         return {
#             "Market Price": price,
#             "Change": change,
#             "Percentage Change": percentage,
#         }
    
#     finally:
#         driver.quit()

# # Example usage:
# stock_data = fetch_real_time_price('AAPL')
# print(stock_data)

#MAIN CODE

# import asyncio
# from pyppeteer import launch

# async def fetch_real_time_price(stock_codes):
#     browser = await launch(headless=True)
#     page = await browser.newPage()
    
#     # Navigate to the URL and wait for the network to be idle
#     await page.goto(f'https://finance.yahoo.com/quote/{stock_code}?guccounter=1', waitUntil='networkidle2')

#     # Function to safely extract text from selectors
#     async def safe_get_text(selector):
#         try:
#             await page.waitForSelector(selector, options={'timeout': 5000})
#             text = await page.evaluate('(selector) => { const element = document.querySelector(selector); return element ? element.innerText : "N/A"; }', selector)
#             return text
#         except Exception as e:
#             print(f"Failed to retrieve {selector}: {str(e)}")
#             return 'N/A'

#     # Define selectors for the elements
#     price_selector = 'fin-streamer[data-field="regularMarketPrice"]'
#     change_selector = 'fin-streamer[data-field="regularMarketChange"]'
#     percentage_change_selector = 'fin-streamer[data-field="regularMarketChangePercent"]'

    # Extract the data
    # price = await safe_get_text(price_selector)
    # change = await safe_get_text(change_selector)
    # percentage_change = await safe_get_text(percentage_change_selector)

    # print({
    #     "Market Price": price,
    #     "Change": change,
    #     "Percentage Change": percentage_change
    # })

import asyncio
from pyppeteer import launch
import pandas as pd

async def fetch_real_time_price(stock_codes):
    browser = await launch(headless=True)
    page = await browser.newPage()

    # Navigate to the URL and wait for the network to be idle
    await page.goto('https://finance.yahoo.com/', waitUntil='networkidle2')

    # Function to safely extract text from selectors
    async def safe_get_text(selector):
        try:
            await page.waitForSelector(selector, options={'timeout': 5000})
            text = await page.evaluate('(selector) => { const element = document.querySelector(selector); return element? element.innerText : "N/A"; }', selector)
            return text
        except Exception as e:
            print(f"Failed to retrieve {selector}: {str(e)}")
            return 'N/A'

    # Define selectors for the elements
    price_selector = 'fin-streamer[data-field="regularMarketPrice"]'
    change_selector = 'fin-streamer[data-field="regularMarketChange"]'
    percentage_change_selector = 'fin-streamer[data-field="regularMarketChangePercent"]'
    

    data = []
    for stock_code in stock_codes:
        await page.goto(f'https://finance.yahoo.com/quote/{stock_code}?guccounter=1', waitUntil='networkidle2')
        price = await safe_get_text(price_selector)
        change = await safe_get_text(change_selector)
        percentage_change = await safe_get_text(percentage_change_selector)
        data.append({
            "Stock Code": stock_code,
            "Market Price": price,
            "Change": change,
            "Percentage Change": percentage_change
        })

    await browser.close()

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    table = df.to_html(index=False)
    #print table
    print(table)


# Run the function
asyncio.get_event_loop().run_until_complete(fetch_real_time_price(['NQ%3DF', 'AAPL', 'GOOGL']))



