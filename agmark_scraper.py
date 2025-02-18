import time
import logging
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

def scrape_agmarknet(state, district, market, commodity, start_date, end_date):
    """
    Scrapes Agmarknet for the given state, district, commodity, and date range.
    """

    base_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    driver = webdriver.Chrome()

    try:
        driver.get(base_url)
        time.sleep(2)  # Allow page to load

        # ✅ Select Commodity
        commodity_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ddlCommodity"))
        )
        Select(commodity_dropdown).select_by_visible_text(commodity)
        time.sleep(2)

        # ✅ Select State
        state_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ddlState"))
        )
        Select(state_dropdown).select_by_visible_text(state)
        time.sleep(3)

        # ✅ Select District
        district_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ddlDistrict"))
        )
        Select(district_dropdown).select_by_visible_text(district)
        time.sleep(3)

        # ✅ Select Market
        market_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ddlMarket"))
        )
        Select(market_dropdown).select_by_visible_text(market)
        time.sleep(3)

        # ✅ Select Date Range (Use txtDate instead of txtDateFrom and txtDateTo)
        try:
            date_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "txtDate"))
            )
            date_input.clear()
            date_input.send_keys(start_date)  # Use start_date to fetch last 3 months' data

        except TimeoutException:
            logging.error("❌ Failed to locate date input field 'txtDate'. Check element IDs.")
            return []

        # ✅ Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "cphBody_GridPriceData"))
        )

        # ✅ Parse Table Data
        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.find_all("tr")
        data_list = [row.get_text(separator="_", strip=True).split("_") for row in rows]

        results = []
        for row in data_list[4:]:  # Skip headers
            if len(row) < 11 or row[4] != commodity:
                continue  # ✅ Skip non-Apple rows

            # ✅ Convert Rs./Quintal to ₹/kg
            min_price = float(row[7]) / 100 if row[7] != '-' else None
            max_price = float(row[8]) / 100 if row[8] != '-' else None
            modal_price = float(row[9]) / 100 if row[9] != '-' else None

            results.append({
                "State": row[0],
                "District": row[1],
                "Market": row[2],
                "Commodity": row[4],
                "Arrivals (Tonnes)": row[6],
                "Min Price (₹/kg)": min_price,
                "Max Price (₹/kg)": max_price,
                "Modal Price (₹/kg)": modal_price,
                "Date": row[10],
            })

        return results

    except Exception as e:
        logging.error(f"❌ Error scraping Agmarknet: {str(e)}")
        return []

    finally:
        driver.quit()
