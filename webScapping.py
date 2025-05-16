from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import os

# === Setup Chrome with visible UI ===
options = Options()
# Comment out below to run in headless mode:
# options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)

# === Launch the site ===
driver.get("https://www.myscheme.gov.in/search")
input("🧑‍💻 Please make the desired changes on the website (e.g., filters). Press ENTER to continue scraping...")

page = 1
seen_links = set()

def save_page_json(schemes, page_num):
    filename = f"page_{page_num}.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(schemes, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(schemes)} schemes to '{filename}'")

def extract_schemes():
    schemes = []
    try:
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.p-4.lg\\:p-8.w-full")))
        cards = driver.find_elements(By.CSS_SELECTOR, "div.p-4.lg\\:p-8.w-full")

        for card in cards:
            try:
                a_tag = card.find_element(By.CSS_SELECTOR, "h2 a")
                name = a_tag.text.strip()
                href = a_tag.get_attribute("href")
                full_link = href if href.startswith("http") else "https://www.myscheme.gov.in" + href

                if full_link not in seen_links:
                    schemes.append({"name": name, "link": full_link})
                    seen_links.add(full_link)
            except:
                continue
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
    return schemes

def click_next_arrow():
    try:
        pagination = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "ul.list-none.flex.flex-wrap.items-center.justify-center")
        ))
        svgs = pagination.find_elements(By.CSS_SELECTOR, "svg.cursor-pointer")

        # The last one is the active Next arrow (enabled)
        for svg in reversed(svgs):
            classes = svg.get_attribute("class")
            if "!cursor-not-allowed" not in classes:
                driver.execute_script("arguments[0].scrollIntoView(true);", svg)
                time.sleep(1)
                svg.click()
                return True

        print("⚠️ No active Next arrow found")
        return False
    except Exception as e:
        print(f"❌ Error clicking next arrow: {e}")
        return False

# === Main Loop ===
while True:
    print(f"\n📄 Scraping Page {page}")
    schemes = extract_schemes()
    save_page_json(schemes, page)

    success = click_next_arrow()
    page += 1
    if not success:
        print("\n⛔ Next page not clickable. Please refresh or fix manually.")
        input("🔄 After fixing the page manually, press ENTER to continue...")
        continue

    
    time.sleep(3)  # Allow next page to load
