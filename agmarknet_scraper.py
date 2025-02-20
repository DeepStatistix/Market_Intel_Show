# file: agmarknet_selenium.py

import json
import time
import logging
from datetime import datetime, timedelta
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

logging.basicConfig(level=logging.INFO)

def market_options_loaded(driver):
    """
    Helper function for WebDriverWait.
    Returns True if the Market dropdown (by ID 'ddlMarket') has more than one option.
    Handles stale element references.
    """
    try:
        dropdown = Select(driver.find_element(By.ID, 'ddlMarket'))
        options = dropdown.options
        return len(options) > 1
    except StaleElementReferenceException:
        return False

def close_popup(driver):
    """
    Attempts to close the ad popup by looking for an element with class 'popup-onload'
    and then clicking its child element with class 'close'.
    """
    try:
        popup = driver.find_element(By.CLASS_NAME, 'popup-onload')
        close_button = popup.find_element(By.CLASS_NAME, 'close')
        close_button.click()
        logging.info("Popup closed")
    except NoSuchElementException:
        logging.info("Popup not found")

def scrape_agmarknet(commodity, state, market):
    """
    Scrape Agmarknet data by selecting the Commodity, State, Market,
    and setting the date to 7 days ago. Waits for each dropdown to update
    before proceeding. Returns a list of dictionaries with the scraped table data.
    """
    initial_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    logging.info(f"Launching Chrome to scrape: commodity={commodity}, state={state}, market={market}")

    # Initialize the browser (use options for headless mode if needed)
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 15)

    try:
        # Open the page
        driver.get(initial_url)
        
        close_popup(driver)
        # --- Step 1: Select Commodity ---
        logging.info("Selecting commodity...")
        commodity_dropdown = Select(wait.until(
            EC.presence_of_element_located((By.ID, 'ddlCommodity'))
        ))
        commodity_dropdown.select_by_visible_text(commodity)
        
        # --- Step 2: Select State ---
        logging.info("Selecting state...")
        state_dropdown = Select(wait.until(
            EC.presence_of_element_located((By.ID, 'ddlState'))
        ))
        state_dropdown.select_by_visible_text(state)
        
        # --- Step 3: Set Date (7 days ago) ---
        logging.info("Setting date to 7 days ago...")
        today = datetime.now()
        desired_date = today - timedelta(days=7)
        date_input = wait.until(EC.presence_of_element_located((By.ID, "txtDate")))
        date_input.clear()
        date_input.send_keys(desired_date.strftime('%d-%b-%Y'))
        
        # --- Step 4: Click first 'Go' button ---
        logging.info("Clicking first 'Go' button...")
        go_button = wait.until(EC.element_to_be_clickable((By.ID, 'btnGo')))
        go_button.click()
        
        # Allow some time for the Market dropdown to update
        time.sleep(3)
        
        # --- Step 5: Wait for Market dropdown to load its options ---
        logging.info("Waiting for Market dropdown to update...")
        wait.until(market_options_loaded)
        
        # --- Step 6: Select Market ---
        logging.info("Selecting market...")
        market_dropdown = Select(driver.find_element(By.ID, 'ddlMarket'))
        market_dropdown.select_by_visible_text(market)
        
        # --- Step 7: Click second 'Go' button ---
        logging.info("Clicking second 'Go' button...")
        go_button = wait.until(EC.element_to_be_clickable((By.ID, 'btnGo')))
        go_button.click()
        
        # --- Step 8: Wait for results table to appear ---
        logging.info("Waiting for results table...")
        wait.until(EC.presence_of_element_located((By.ID, 'cphBody_GridPriceData')))
        
        # --- Step 9: Parse the page using BeautifulSoup ---
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find("table", {"id": "cphBody_GridPriceData"})
        if not table:
            logging.error("Results table not found in page source.")
            return []
        
        rows = table.find_all("tr")
        logging.info(f"Found {len(rows)} rows in the table.")

        jsonList = []
        for row in rows[1:]:  # skip header
            cols = row.find_all("td")
            # Check for at least 11 columns
            if len(cols) < 9:
                continue
            data_dict = {
                "S.No": cols[0].get_text(strip=True),
                "District": cols[1].get_text(strip=True),
                "Market": cols[2].get_text(strip=True),
                "Commodity": cols[3].get_text(strip=True),
                "Variety": cols[4].get_text(strip=True),
                "Grade" : cols[5].get_text(strip=True),
                "Min Price": cols[6].get_text(strip=True),
                "Max Price": cols[7].get_text(strip=True),
                "Model Price": cols[8].get_text(strip=True),
                "Date": cols[9].get_text(strip=True)
            }

            jsonList.append(data_dict)
        
        logging.info(f"Scraped {len(jsonList)} rows from table.")
        return jsonList

    except Exception as e:
        logging.error(f"Error during scraping: {str(e)}")
        return []
    finally:
        driver.quit()

if __name__ == "__main__":
    test_params = [
        ("Apple", "NCT of Delhi", "Azadpur"),
        ("Apple", "Karnataka", "Bangalore"),
        ("Apple", "Uttar Pradesh", "Lucknow"),
        ("Apple", "West Bengal", "Mechua"),
        ("Apple", "Jammu and Kashmir", "Narwal Jammu (F&V)")
    ]
    
    all_data = []
    for commodity, state, market in test_params:
        print(f"\nScraping data for Commodity: {commodity}, State: {state}, Market: {market}")
        data = scrape_agmarknet(commodity, state, market)
        all_data.extend(data)
        print("Scraped Data:")
        for row in data:
            print(row)
    
    if all_data:
        # Ensure the output directory exists
        output_dir = "data/scraped"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        import pandas as pd
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(output_dir, "apple_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    else:
        print("No data scraped.")