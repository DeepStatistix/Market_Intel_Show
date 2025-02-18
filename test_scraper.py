from agmark_scraper import scrape_agmarknet
import logging
from datetime import datetime, timedelta

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define test parameters
state = "West Bengal"
district = "Kolkata"
market = "Mechua"
commodity = "Apple"

# ✅ Set Date Range (Last 30 Days)
start_date = (datetime.now() - timedelta(days=30)).strftime('%d-%b-%Y')
end_date = datetime.now().strftime('%d-%b-%Y')

logging.info(f"🔍 Testing Agmarknet Scraper for: State='{state}', District='{district}', Market='{market}', Date Range: {start_date} - {end_date}")

# ✅ Run Scraper with Error Handling
try:
    scraped_data = scrape_agmarknet(state, district, market, commodity, start_date, end_date)

    # ✅ Check if data was successfully scraped
    if scraped_data:
        logging.info(f"✅ Successfully scraped {len(scraped_data)} records!")
        for entry in scraped_data:
            print(entry)
    else:
        logging.error("❌ No data returned. Check website, parameters, or scraper logic.")

except Exception as e:
    logging.error(f"❌ Unexpected error occurred: {str(e)}")
