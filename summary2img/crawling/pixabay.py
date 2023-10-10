import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import Request, urlretrieve, URLopener
from time import sleep
import os

# prepare for crawling

def prepare(driver,theme):
    url = f"https://pixabay.com/ko/images/search/{theme}"  # 한국어 검색어에 대한 url 생성
    print(url)
    
    driver.get(url=url)

    folder_name = "./img_data"  # 상위 폴더(img_data) 생성
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    folder_name = f"./img_data/{theme}"  # 하위 폴더(theme) 생성
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# get image url
def get_img_url(driver):
    try:
        images = driver.find_elements(
            by=By.CSS_SELECTOR, value="#app > div:nth-child(1) > div > div.container--wYO8e > div.results--mB75j > div > div > div > div > div > a > img"
        )
    except:
        print("드라이버 css 찾기 실패.")
        
    img_url = []
    for image in images:
        try:
            url = image.get_attribute("src")
            img_url.append(url)
        except:
            print(f"{image}_src 얻기 실패.")
    return img_url

# get png file
def get_png_file(driver,theme):
    folder_name=f"./img_data/{theme}"
    
    page_num = driver.find_elements(By.CSS_SELECTOR, "#app > div:nth-child(1) > div > div.container--wYO8e > div.pagination--t1UWv > div.pages--1CAfr > div")
    page_num = page_num[-1].text[2:]

    headers={"User-Agent": "Mozilla/5.0"}

    for i in range(1, int(page_num)+1):
        img_url = get_img_url(driver)
        print(f"{i}_page에서 찾은 이미지 개수 : {len(img_url)}")
        for link in img_url :
            file_name = link.split('/')[-1].split('.')[-2]
            os.system(f"curl {link} > {folder_name}/{file_name}.png")
        driver.find_element(by=By.CSS_SELECTOR, value="#app > div:nth-child(1) > div > div.container--wYO8e > div.pagination--t1UWv > div.pages--1CAfr > a").click() # 다음 페이지 이동 버튼 클릭
        sleep(3)

def main():
    driver = webdriver.Chrome()
    theme = input("검색어를 입력하세요: ")
    prepare(driver,theme)
    get_png_file(driver,theme)

if __name__=="__main__":
    main()