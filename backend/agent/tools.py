from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import urllib.parse

def parse_html(html: str) -> list[dict]:
    print(f"Parsing HTML")
    soup = BeautifulSoup(html, "html.parser")
    
    # Debug: Print page title to see what page we're on
    title = soup.find('title')
    if title:
        print(f"Page title: {title.get_text()}")
    
    results = []
    
    # Look for DoorDash store links specifically (based on the URL pattern provided)
    # Pattern: /store/{restaurant-name}-{location}-{id}/{store-id}/
    store_links = soup.find_all("a", href=True)
    
    for link in store_links:
        href = link.get('href', '')
        if isinstance(href, str) and '/store/' in href:
            text = link.get_text(strip=True)
            if text and len(text) > 3:
                # Clean up the link to get the base store URL
                if '?' in href:
                    href = href.split('?')[0]  # Remove query parameters
                if '#' in href:
                    href = href.split('#')[0]  # Remove fragments
                
                # Make sure it's a full URL
                if not href.startswith('http'):
                    href = f"https://www.doordash.com{href}"
                
                results.append({
                    "text": text,
                    "link": href,
                    "type": "restaurant"
                })
                print(f"Found restaurant: {text} -> {href}")
    
    # If no store links found, look for any links that might be restaurants
    if not results:
        print("No store links found, looking for restaurant-like links...")
        restaurant_keywords = ['pizza', 'restaurant', 'food', 'delivery', 'takeout', 'dine']
        
        for link in store_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            if text and len(text) > 3 and not text.lower() in ['doordash', 'back home', 'home', 'search']:
                # Check if the text contains restaurant-related keywords
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in restaurant_keywords):
                    if isinstance(href, str) and not href.startswith('http'):
                        href = f"https://www.doordash.com{href}"
                    
                    results.append({
                        "text": text,
                        "link": href,
                        "type": "potential_restaurant"
                    })
                    print(f"Found potential restaurant: {text} -> {href}")
    
    # Debug: Print summary
    print(f"Total restaurant results found: {len(results)}")
    if len(results) > 0:
        print("Restaurant results:")
        for i, result in enumerate(results[:10]):  # Show first 10
            print(f"  {i+1}. {result['text']} -> {result['link']}")
    
    return results


def search_doordash(url: str, search_term: str, location: str) -> list[dict]:
    print(f"Searching {url} for {search_term} in {location}")
    
    # Configure Chrome options to avoid detection
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    
    # Remove webdriver property to avoid detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    try:
        # Try different DoorDash search URL patterns
        search_urls = [
            f"https://www.doordash.com/search?query={urllib.parse.quote(search_term)}&location={urllib.parse.quote(location)}",
            f"https://www.doordash.com/search?query={urllib.parse.quote(f'{search_term} {location}')}",
            f"https://www.doordash.com/food-delivery/{urllib.parse.quote(location)}-restaurants/",
            f"https://www.doordash.com/food-delivery/{urllib.parse.quote(location)}-food-delivery/"
        ]
        
        for i, search_url in enumerate(search_urls):
            print(f"Trying search URL {i+1}: {search_url}")
            driver.get(search_url)
            time.sleep(8)  # Wait for search results
            
            page_html = driver.page_source
            parsed_html = parse_html(page_html)
            
            if len(parsed_html) > 2:  # More than just navigation links
                print(f"Found {len(parsed_html)} links via URL {i+1}")
                return parsed_html
        
        # If direct URLs don't work, try the form-based approach
        print("Direct URLs didn't work, trying form-based search...")
        driver.get(url)
        print("Page loaded, waiting for elements...")
        
        # Wait for page to load with explicit wait
        wait = WebDriverWait(driver, 20)
        
        # Try multiple selectors for the search input
        search_selectors = [
            '//input[contains(@placeholder, "Search")]',
            '//input[contains(@placeholder, "search")]',
            '//input[@type="text"]',
            '//input[contains(@class, "search")]',
            '//input[contains(@id, "search")]',
            '//input[contains(@name, "search")]',
            '//input[contains(@aria-label, "Search")]',
            '//input[contains(@aria-label, "search")]'
        ]
        
        search_food = None
        for selector in search_selectors:
            try:
                search_food = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                print(f"Found search input with selector: {selector}")
                break
            except TimeoutException:
                continue
        
        if not search_food:
            # If no search input found, try to get the page content anyway
            print("No search input found, getting page content...")
            page_html = driver.page_source
            return parse_html(page_html)
        
        # Clear the input and enter search terms
        search_food.clear()
        search_food.send_keys(f"{search_term} {location}")
        time.sleep(2)
        search_food.send_keys(Keys.ENTER)
        
        print("Search submitted, waiting for results...")
        time.sleep(8)  # Wait for search results to load
        
        page_html = driver.page_source
        parsed_html = parse_html(page_html)
        
        print(f"Found {len(parsed_html)} links")
        return parsed_html
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        # Return empty results on error
        return []
    finally:
        driver.quit()