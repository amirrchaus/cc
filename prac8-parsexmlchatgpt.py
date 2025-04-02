import csv
import requests
import xml.etree.ElementTree as ET

def loadRSS(url, filename):
    """Fetches the RSS feed from the given URL and saves it as an XML file."""
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # Check if the request was successful

        with open(filename, 'wb') as f:
            f.write(resp.content)

        print(f"RSS feed loaded and saved to '{filename}'.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching RSS feed: {e}")

def parseXML(xmlfile):
    """Parses the given XML file and extracts news items."""
    try:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        newsitems = []

        for item in root.findall('./*/item'):
            news = {}
            for child in item:
                if child.tag.endswith('thumbnail'):
                    continue
                if child.tag.endswith('content'):
                    news['media'] = child.attrib.get('url', '')
                else:
                    news[child.tag] = child.text
            newsitems.append(news)

        return newsitems
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []

def savetoCSV(newsitems, filename):
    """Saves news items to a CSV file."""
    fields = ['guid', 'title', 'pubDate', 'description', 'link', 'media']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(newsitems)

    print(f"Data saved to {filename}.")

def main():
    # Choose RSS source
    sources = {
        "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
        "Hindustan Times": "https://www.hindustantimes.com/rss.xml"
    }

    print("Choose a news source:")
    for i, source in enumerate(sources.keys(), 1):
        print(f"{i}. {source}")

    choice = int(input("Enter choice (1 or 2): "))
    source_name = list(sources.keys())[choice - 1]
    source_url = sources[source_name]

    xml_filename = f"{source_name.replace(' ', '_').lower()}.xml"
    csv_filename = f"{source_name.replace(' ', '_').lower()}.csv"

    loadRSS(source_url, xml_filename)
    newsitems = parseXML(xml_filename)

    if newsitems:
        savetoCSV(newsitems, csv_filename)

if __name__ == "__main__":
    main()
