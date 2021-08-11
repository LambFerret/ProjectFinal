from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager #크롬드라이버 경로지정 없이 하는거 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import time
import ssl


from urllib.request import urlretrieve
import tkinter, tkinter.constants, tkinter.filedialog

def askDialog():
    return tkinter.filedialog.askdirectory()

def inp(text):
    return input(text)

ssl._create_default_https_context = ssl._create_unverified_context

def imagescrape():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        driver.maximize_window()
        for i in range(1, searchPage + 1):
            url = "https://www.shutterstock.com/search?searchterm=" + searchTerm + "&sort=popular&image_type=" + image_type + "&search_source=base_landing_page&language=en&page=" + str(i)
            driver.get(url)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(4)                                                           
            data = driver.execute_script("return document.documentElement.outerHTML")
            print("Page " + str(i))
            scraper = BeautifulSoup(data, "lxml")
            img_container = scraper.find_all("img", {"class":"z_h_9d80b z_h_2f2f0"})
            for j in range(0, len(img_container)-1):
                img_src = img_container[j].get("src")
                name = img_src.rsplit("/", 1)[-1]
                try:
                    urlretrieve(img_src, os.path.join(scrape_directory, os.path.basename(img_src)))
                    print("Scraped " + name)
                except Exception as e:
                    print(e)
        driver.close()
    except Exception as e:
        print(e)

while True:
    while True:
        print("폴더 경로 선택")
        scrape_directory = askDialog()
        break
    while True:
        image_type = 'photo'
        break
    while True:
            searchTerm = inp("검색어: ")
            break
    while True:
        searchPage = int(input("몇 페이지: "))
        break
    imagescrape()
    print("완료")
    
