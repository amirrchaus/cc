import csv
import requests
import xml.etree.ElementTree as ET

def loadRSS():
    url = 'http://feeds.bbci.co.uk/news/rss.xml'
    resp = requests.get(url)
    with open('topnewsfeed.xml', 'wb') as f:
        f.write(resp.content)
    print("RSS loaded and saved to 'topnewsfeed.xml'.")

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    print("XML file parsed successfully.")
    root = tree.getroot()
    newsitems = []

    for item in root.findall('./*/item'):
        print("Processing item...")
        news = {}
        for child in item:
            if child.tag.endswith('thumbnail'):
                continue
            if child.tag.endswith('content'):
                news['media'] = child.attrib.get('url', '')
            else:
                news[child.tag] = child.text
        print("Item processed:", news)
        newsitems.append(news)

    return newsitems

def savetoCSV(newsitems, filename):
    fields = ['guid', 'title', 'pubDate', 'description', 'link', 'media', '{http://search.yahoo.com/mrss/}thumbnail']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(newsitems)

    print(f"Data saved to {filename}.")

def main():
    loadRSS()
    newsitems = parseXML('topnewsfeed.xml')
    print("Parsing XML completed.")
    savetoCSV(newsitems, 'topnews.csv')

if __name__ == "__main__":
    main()
