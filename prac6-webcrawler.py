import requests
from bs4 import BeautifulSoup


def crawl_website(url, word):
    try:
        # Send a GET request to the website
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all anchor tags
    links = soup.find_all("a")

    # Initialize counters and storage
    total_links = 0
    matching_links = []

    # Process each link
    for link in links:
        href = link.get("href")
        total_links += 1
        if href and word.lower() in href.lower():
            matching_links.append(href)

    # Remove duplicates from matching links
    unique_matching_links = list(set(matching_links))

    # Output results
    print("=== Crawler Results ===")
    print(f"Total links scanned: {total_links}")
    print(f"Links containing the word '{word}': {len(unique_matching_links)}")
    print("\nMatching links:")
    for idx, link in enumerate(unique_matching_links, 1):
        print(f"{idx}. {link}")


if __name__ == "__main__":
    # User inputs
    website_url = "https://facebook.com"
    search_word = "login"

    crawl_website(website_url, search_word)
