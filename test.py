import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Niacinamide"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
content = soup.find("div", {"id": "mw-content-text"})
print(content.get_text()[:500])