from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import time
import requests

def download_adopted_dog_images(start_page=1, end_page=5678):
    save_dir = 'adopted_dogs'
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
            url = f"https://www.animal.go.kr/front/awtis/public/publicAllList.do?totalCount=67868&pageSize=12&menuNo=1000000064&searchSDate=2024-01-01&searchEDate=2025-01-21&searchUpKindCd=417000&page={page}"
            print(f"페이지 {page} 처리 중...")
            driver.get(url)
            time.sleep(3)
            
            # 상태가 '종료(입양)'인 이미지를 찾기 위한 XPath
            status_elements = driver.find_elements(
                By.XPATH,
                "//*[@id='contents']/div/ul/li/a/ol/li[2]/div[5]/div[2][contains(text(), '종료(입양)')]"
            )
            
            for status_element in status_elements:
                try:
                    # 상위 요소로 이동하여 이미지 찾기
                    parent = status_element.find_element(By.XPATH, "./ancestor::a")
                    img = parent.find_element(By.TAG_NAME, "img")
                    
                    if img:
                        img_url = img.get_attribute('src')
                        if not img_url:
                            continue 
                        
                        if not img_url.endswith('no_image.gif'):
                            img_name = f"adopted_dog_page{page}_{total_images}.jpg"
                            img_path = os.path.join('adopted_dogs', img_name)
                            
                            # 이미지 다운로드
                            img_data = requests.get(img_url, timeout=10).content
                            with open(img_path, 'wb') as img_file:
                                img_file.write(img_data)
                            
                            total_images += 1
                            print(f"이미지 다운로드 완료: {img_name}")
                            print(f"상태: 종료(입양)")
                            print(f"이미지 URL: {img_url}")
                                                
                except Exception as e:
                    continue
            
            time.sleep(1)       
            
    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}") 

    finally:
        driver.quit()
        print(f"크롤링 완료. 총 {total_images}개의 이미지를 다운로드했습니다.")

# 실행 
download_adopted_dog_images(1, 3)