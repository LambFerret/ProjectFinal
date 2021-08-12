#import selenium drivers
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException  

#import helper libraries
import time
import os

from bs4 import BeautifulSoup
from urllib.request import urlretrieve

#custom patch libraries
import patch 


class ShutterstockImageScraper():
    def __init__(self,webdriver_path,scrape_directory, searchTerm="cat", image_type='photo', searchPage=1,headless=False,min_resolution=(0,0),max_resolution=(1920,1080)):
        #check parameter types
        if (type(searchPage)!=int):
            print("[Error] Number of images must be integer value.")
            return
        if not os.path.exists(scrape_directory):
            print("[INFO] Image path not found. Creating a new folder.")
            os.makedirs(scrape_directory)
        #check if chromedriver is updated
        while(True):
            try:
                #try going to www.google.com
                options = Options()
                if(headless):
                    options.add_argument('--headless')
                driver = webdriver.Chrome(webdriver_path, chrome_options=options)
                driver.maximize_window()
                driver.get("https://www.google.com")
                break
            except:
                #patch chromedriver if not available or outdated
                try:
                    driver
                except NameError:
                    is_patched = patch.download_lastest_chromedriver()
                else:
                    is_patched = patch.download_lastest_chromedriver(driver.capabilities['version'])
                if (not is_patched): 
                    print("[WARN] Please update the chromedriver.exe in the webdriver folder according to your chrome version:https://chromedriver.chromium.org/downloads")
                    break
        self.driver = driver
        self.searchTerm = searchTerm
        self.searchPage = searchPage
        self.webdriver_path = webdriver_path
        self.scrape_directory = scrape_directory
        
        self.headless=headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.saved_extension = "jpg"
        self.valid_extensions = ["jpg","png","jpeg"]
        self.image_type = image_type
    
    def find_save_image(self):
        print("[INFO] Scraping for image link... Please wait.")
        for i in range(1, self.searchPage + 1):
            url = "https://www.shutterstock.com/search?searchterm=" + self.searchTerm + "&sort=popular&image_type=" + self.image_type + "&search_source=base_landing_page&language=en&page=" + str(i)
            self.driver.get(url)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(4)                                                           
            data = self.driver.execute_script("return document.documentElement.outerHTML")
            print("Page " + str(i))
            scraper = BeautifulSoup(data, "lxml")
            img_container = scraper.find_all("img", {"class":"z_h_9d80b z_h_2f2f0"})
            for j in range(0, len(img_container)-1):
                img_src = img_container[j].get("src")
                name = img_src.rsplit("/", 1)[-1]
                try:
                    urlretrieve(img_src, os.path.join(self.scrape_directory, os.path.basename(img_src)))
                    print("Scraped " + name)
                except Exception as e:
                    print(e)
        self.driver.close()
        print("[INFO] Download Completed. Please note that some photos are not downloaded as it is not in the right format (e.g. jpg, jpeg, png)")
        
        
        







