from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import time
import requests

def download_waiting_dog_images(start_page=1, end_page=3):
    save_dir = 'wait_adoption_img_sample'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 크롬 드라이버 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)
    total_images = 0
    
    try:
        for page in range(start_page, end_page + 1):
            url = f"https://www.animal.go.kr/front/awtis/protection/protectionList.do?totalCount=10766&pageSize=10&boardId=&desertionNo=&menuNo=1000000060&searchSDate=2024-01-01&searchEDate=2025-01-23&searchUprCd=&searchOrgCd=&searchCareRegNo=&searchUpKindCd=417000&searchKindCd=&searchSexCd=&searchRfid=&&page={page}"
            print(f"페이지 {page} 처리 중...")
            driver.get(url)
            time.sleep(3)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            div_tags = soup.find_all("div", class_="inner-img") 
            
            for i, div_tag in enumerate(div_tags):
                style_attr = div_tag.get("style", "")  # style 속성에서 background-image URL 추출 
                if "background-image" in style_attr:
                    url_start = style_attr.find("url(") + 4
                    url_end = style_attr.find(")", url_start)
                    image_url = style_attr[url_start:url_end].strip("'\"")
                
                    base_url = "https://www.animal.go.kr"
                    if not image_url.startswith("http"):
                        image_url = base_url + image_url
                    
                    try:
                        img_response = requests.get(image_url, timeout=10)
                        if img_response.status_code == 200:
                            file_name = os.path.join(save_dir, f"page_{page}_image_{i+1}.jpg")
                            with open(file_name, "wb") as file:
                                file.write(img_response.content)
                            print(f"이미지 저장 완료: {file_name}")
                            total_images += 1
                        else:
                            print(f"이미지 다운로드 실패: {image_url}")
                    except Exception as e:
                        print(f"이미지 다운로드 중 오류 발생: {e}")
                else:
                    print(f"style 속성에 background-image가 없습니다: {div_tag}")
            
            time.sleep(1)
    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")
    finally:
        driver.quit()
        print(f"크롤링 완료. 총 {total_images}개의 이미지를 다운로드했습니다.")

# 실행 
download_waiting_dog_images(1, 3) 