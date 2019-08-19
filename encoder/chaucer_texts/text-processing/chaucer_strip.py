from bs4 import BeautifulSoup
import requests
import pprint as pp
import csv
import re
import os

# returns if a line is a translated line
def lineRelevant(line):
    return not(re.match("____________________________", line)) and not("heere biygnneth" in line) and not ("heere endeth" in line)

def getLinksFromChaucerTable(url):
    processLink(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    for liTag in soup.find("div", {"id": "content-panels"}).find_all("li"):
        for aTag in liTag.find_all("a", href=re.compile("http://hwpi.harvard.edu/chaucer/pages")):
            links.append(aTag["href"])
    del links[-1]
    return links

# function that takes a link to a Chaucer fragment and converts it to expected format
def processLink(url):
    file = open("chaucer_texts/linebyline/" + os.path.basename(url) + "linebyline.txt", "w+")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    textSpans = soup.find_all("span", {"style": "font-family:'book antiqua', palatino"})
    i = 1
    lineMerge = ""
    for line in textSpans:
        text = line.get_text()
        result = ''.join([i for i in text if not i.isdigit()]).strip() + "\n"
        file.write(result)

links = getLinksFromChaucerTable("https://chaucer.fas.harvard.edu/pages/text-and-translations")
for link in links:
    print ("Processing " + link + "...")
    processLink(link)
