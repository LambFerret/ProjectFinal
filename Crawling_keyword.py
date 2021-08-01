from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os

def Search_this(chromedriver_path, input_keyword, crawl_num):
    save_folder = chromedriver_path+"/"+input_keyword
    try: 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    except OSError:
        print("cant make folder")
                    
    driver = webdriver.Chrome(chromedriver_path +"/" +'chromedriver') 
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
    elem = driver.find_element_by_name("q")
    elem.send_keys(input_keyword)
    elem.send_keys(Keys.RETURN)
     
    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.body.scrollHeight")
     
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
     
        time.sleep(SCROLL_PAUSE_TIME)
     
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height
    images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
    count = 1
    for image in images:
        try: 
            image.click()
            time.sleep(2)
            imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute("src")
            urllib.request.urlretrieve(imgUrl, save_folder +"/"+str(count) + ".jpg")
            if count == crawl_num:
                break
            else:
                count = count + 1
        except:
            pass
    driver.close()
    
Search_this("C:/Users/LambFerret/Python/FINAL PROJECT", "winter wallpaper", 3)
