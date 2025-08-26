import pandas as pd 
import numpy as np
import re  

def preprocessed_df(csv_path, encoding, base_year):
    df = pd.read_csv(csv_path, encoding=encoding)

    drop_cols = ['공고번호', '동물등록번호', '발생장소', '보호장소', '접수일시', '보호센터']
    df = df.drop(columns=drop_cols)

    str_cols = ["품종", "성별", "관할기관", "중성화여부", "나이", "체중"]
    for col in str_cols:
      df[col] = df[col].astype(str).str.strip() 

    # 품종 
    rename_map = {
        "비숑 프리제": "비숑프리제",
        "리트리버믹스견": "리트리버믹스",
        "레온베르거믹스인듯": "레온베르거믹스",
        "빠삐용믹스인듯": "빠삐용믹스",
        "화이트테리어믹스인듯": "화이트테리어믹스",
        "웰시 코기 펨브로크": "웰시코기",
        "카네 코르소": "카네코르소",
        "미니어쳐 핀셔": "미니핀",
        "마리노이즈": "말리노이즈",
        "풍산견": "풍산개",
        "토이 푸들": "푸들",
        "말티푸+요키": "말티푸",
    }
    df["품종"] = df["품종"].replace(rename_map) 
    korean_breeds = {"진돗개","풍산개","삽살개"}
    foreign_breeds = {"푸들","말티즈","말티푸","비글","포메라니안","빠삐용믹스","화이트테리어믹스",
                      "말리노이즈","시츄","리트리버믹스","비숑프리제","레온베르거믹스",
                      "웰시코기","스피츠","카네코르소","미니핀","치와와"} 
    conds = [df["품종"].str.contains("믹스", na=False),
            df["품종"].isin(foreign_breeds),
            df["품종"].isin(korean_breeds)]
    choices = [0,1,2]
    df["품종_enc"] = np.select(conds, choices, default=0).astype("int8") 

    # 성별 
    gender_map = {"수컷":0,"암컷":1,"미상":2}
    df["성별_enc"] = df["성별"].map(gender_map).astype("int8") 

    # 관할기관 
    s = df["관할기관"].astype(str).str.replace("\u00A0", " ", regex=False)
    root = s.str.extract(
        r'^(서울|인천|경기|전라|전북|전남|광주|경상|경북|경남|부산|대구|울산|충청|충북|충남|대전|세종|강원|제주)')[0]
    region_map = {
        "서울":0, "인천":0, "경기":0,
        "전라":1, "전북":1, "전남":1, "광주":1,
        "경상":2, "경북":2, "경남":2, "부산":2, "대구":2, "울산":2,
        "충청":3, "충북":3, "충남":3, "대전":3, "세종":3,
        "강원":4,
        "제주":5,
    }
    df["관할기관_enc"] = root.map(region_map).astype("int8") 

    # 중성화여부 
    neuter_map = {"아니오":0, "예":1, "미상":2}
    df["중성화여부_enc"] = df["중성화여부"].map(neuter_map).astype("int8")

    # 나이 
    df["출생연도"] = df["나이"].str.extract(r"((19|20)\d{2})")[0].astype(float)
    df["age_cal"] = base_year-df["출생연도"]
    df["나이_enc"] = pd.cut(df["age_cal"], bins=[-np.inf, 2, 5, np.inf], labels=[0, 1, 2]).astype("int8")

    # 체중
    df["체중"] = df["체중"].str.extract(r"(\d+(\.\d+)?)")[0].astype(float)
    df["체중_enc"] = pd.cut(df["체중"], bins=[-np.inf,4,14,np.inf], labels=[0,1,2]).astype("int8") 

    selected_df = df[["품종_enc","성별_enc","관할기관_enc","중성화여부_enc","나이_enc","체중_enc", "색상","배경","귀 모양","악세서리","입 모양","상태(입양)"]]
    rename_col = {
        "품종_enc": "breed",
        "성별_enc": "gender",
        "관할기관_enc": "region",
        "중성화여부_enc": "neuter_status",
        "나이_enc": "age",
        "체중_enc": "weight",
        "색상": "color",
        "배경": "background",
        "귀 모양": "ear_shape",
        "악세서리": "accessory",
        "입 모양": "mouth_shape",
        "상태(입양)": "adoption_status",
    }
    final_df = selected_df.rename(columns=rename_col)

    return final_df 