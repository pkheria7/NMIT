from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pprint
import time
import json
import os

def to_camel_case(text):
    # Split the text into words
    words = text.split()
    # Convert first word to lowercase
    result = words[0].lower()
    # Capitalize first letter of subsequent words and append
    for word in words[1:]:
        result += word.capitalize()
    return result

def scrape_scheme_info(driver, wait, url, name):
    driver.get(url)
    # input(f"Please review and reload the page if needed for: {url}. Press Enter to continue...")
    
    # Helper to extract section text
    def get_section_by_heading(heading_text):
        try:
            heading = wait.until(EC.presence_of_element_located((By.XPATH, f"//h3[normalize-space()='{heading_text}']")))
            container = heading.find_element(By.XPATH, "./ancestor::div[contains(@class,'pt-10')]//following-sibling::div")
            return container.text.strip()
        except:
            return None

    # Wait for h1 to load and get scheme name
    try:
        h1 = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        scraped_scheme_name = h1.text.strip()
    except:
        scraped_scheme_name = ""

    # Scrape all sections
    data = {
        "schemeName": name,
        "details": get_section_by_heading("Details"),
        "benefits": get_section_by_heading("Benefits"),
        "eligibility": get_section_by_heading("Eligibility"),
        "applicationProcess": get_section_by_heading("Application Process"),
        "documentsRequired": get_section_by_heading("Documents Required"),
        "sourcesAndReferences": {}
    }

    # Get all references
    try:
        sources_container = driver.find_element(By.ID, "sources")
        links = sources_container.find_elements(By.TAG_NAME, "a")
        for link in links:
            label = link.text.strip()
            href = link.get_attribute("href")
            if label and href:
                camel_case_label = to_camel_case(label)
                data["sourcesAndReferences"][camel_case_label] = href
    except:
        pass

    return data

def process_page(page_number, driver, wait):
    input_file = f'page_{page_number}.json'
    output_file = f'OutputJson/page_{page_number}_info.json'
    
    # Create output directory if it doesn't exist
    os.makedirs('OutputJson', exist_ok=True)
    
    # Read input schemes
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_schemes = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} file not found!")
        return False
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file!")
        return False

    all_schemes_data = []

    try:
        # Process each scheme
        for scheme in input_schemes:
            original_name = scheme["name"]
            scheme_url = scheme["link"]
            print(f"\nProcessing scheme: {original_name}")
            try:
                scheme_data = scrape_scheme_info(driver, wait, scheme_url, original_name)
                scheme_data["Original Link"] = scheme_url
                all_schemes_data.append(scheme_data)
                print(f"Successfully scraped: {original_name}")
            except Exception as e:
                print(f"Error processing {original_name}: {str(e)}")
                all_schemes_data.append({
                    "schemeName": original_name,
                    "Original Link": scheme_url,
                    "error": str(e)
                })
            
            # Add a small delay between requests
            time.sleep(2)

        # Save data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_schemes_data, f, indent=4, ensure_ascii=False)
        print(f"\nData has been saved to '{output_file}'")
        return True

    except Exception as e:
        print(f"Error processing page {page_number}: {str(e)}")
        return False

def main():
    # Setup
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # comment this if you want to see the browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    try:
        # Process all pages from 1 to 344
        for page_num in range(1, 8):
            print(f"\nProcessing page {page_num}...")
            success = process_page(page_num, driver, wait)
            if not success:
                print(f"Failed to process page {page_num}")
            
            # Add a small delay between pages
            time.sleep(3)

    finally:
        # Close driver
        driver.quit()

if __name__ == "__main__":
    main()