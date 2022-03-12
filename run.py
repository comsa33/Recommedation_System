import json
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import textdistance
import time
from glob import glob
import os

from preprocess import Preprocess

class SEARCH_RECOMMEND:
    
    def __init__(self, df, qval=None, style_ths=0.1):

        self.style_ths_str = str(style_ths)[-1]
        self.df = df
        self.qval = qval  # Q-Value : 거리 유사도 알고리즘에서 q-gram 으로 문장을 어떻게 나눌지 결정하는 파라미터 (None:단어기반, 1:철자기반, 2~more:q-gram)
        self.df_ = self.preprocess_df(self.df)  # new_tag feature를 새롭게 추가 전처리
        self.user_item_set = set()  # 사용자가 아이템을 선택(검색/추천)시 => 사용자가 선택한 아이템 목록
        self.user_item_len = 0
        self.cold_start = True   # Cold Start : 최적화된 추천 알고리즘 선택을 위해 사용자의 아이템 목록이 비어있는지 확인하는 변수
        
    def preprocess_df(self, df):
        # make new-tag
        # top3_style, tags, name, projectId 를 하나의 string으로 만들기
        df['new_tag'] = list(
                            zip(
                                df[f'top_style_{self.style_ths_str}'].tolist(), 
                                df['tags'].tolist(),
                                df['name'].tolist(),
                                df['projectId'].tolist()
                                )
                            )
        df['new_tag'] = df['new_tag'].apply(self.reduce_newtag)
        df['new_tag'] = df['new_tag'].apply(lambda x: ' '.join(x).lower())
        return df
    
    def reduce_newtag(self, x):
        newtag = []
        for tag in x:
            if type(tag) == str:
                newtag.append(tag)
            else:
                newtag.extend(tag)
        return newtag
    
    # 사용자 셋에 아이템 추가
    def add_item_in_user_item_set(self, prod):
        prev_len = len(self.user_item_set)
        print(f'사용자 님의 아이템 목록에 [{prod}]를 추가합니다.')
        id_ = self.df_[self.df_['name'] == prod]['product_id'].values[0]
        cat_ = self.df_[self.df_['name'] == prod]['category'].values[0]
        self.user_item_set.add((id_, cat_))
        self.cold_start = False
        print(f'사용자 님이 현재 선택한 아이템 목록 : {self.user_item_set}')
        if len(self.user_item_set) > prev_len:
            self.user_item_len += 1
        return id_
    
    # 사용자 셋에서 아이템 제거
    def remove_item_from_user_item_set(self, prod):
        prev_len = len(self.user_item_set)
        print(f'사용자 님의 아이템 목록에서 [{prod}]를 제거합니다.')
        id_ = self.df_[self.df_['name'] == prod]['product_id'].values[0]
        cat_ = self.df_[self.df_['name'] == prod]['category'].values[0]
        self.user_item_set.remove((id_, cat_))
        print(f'사용자 님의 현재 남아있는 선택된 아이템 목록 : {self.user_item_set}')
        # 사용자 목록이 비게 되면 콜드스타트
        if len(self.user_item_set) > 1:
            self.cold_start = True
        if len(self.user_item_set) < prev_len:
            self.user_item_len -= 1
            
    # 사용자가 선택한 아이템과 데이터 내 모든 아이템들과의 유사도 계산
    def get_similarity_score(self, search_prod_tag, algo='sorensen'):
        """
        search_prod_tag : string, 사용자가 선택한 아이템 이름으로부터 매칭된 self.df_['new_tag'].values
        """
        # calculate similarity
        sim_score = []
        new_tag = self.df_['new_tag'].tolist()
        start_time = time.time()
        for i, tag in enumerate(new_tag):
            if algo == 'sorensen':
                sim_score.append(textdistance.Sorensen(qval=self.qval, as_set=True).normalized_similarity(search_prod_tag, tag))
            elif algo == 'ncd':
                sim_score.append(textdistance.EntropyNCD(qval=self.qval).normalized_similarity(search_prod_tag, tag))
        print(f'검색 태그 : {search_prod_tag}')
        print(f'검색 시간 : {round(time.time()-start_time, 4)}초\t검색 알고리즘 : {algo}')
        print()
        return sim_score
    
    # 사용자 선택한 아이템/아이템 목록으로부터 유사도를 통해 아이템 추천
    def search_product(self, prod, topn=10, algo='sorensen'):
        """
        prod : string, 아이템 이름
        topn : int, 유사도 상위 n개의 아이템 추천
        algo : 'sorensen' - 토큰 기반 거리 유사도 알고리즘, 'ncd' - 문장 압축을 통한 유사도 알고리즘
        """
        # 아이템을 사용자 아이템 목록에 추가
        # declare id, category of searching product
        id_ = self.add_item_in_user_item_set(prod)
        prod_cat = self.df_[self.df_['product_id'] == id_]['category'].values[0]
        
        print(f"검색 아이템이 해당한 프로젝트 ID : {self.df_[self.df_['name'] == prod]['projectId'].values[0]}")
        
        if self.cold_start:
            # retrieve the product tag from the input product id
            # 동일 아이템이 여러 전문가에 사용될 경우 => awesome_score 가 더 높은 점수를 받은 아이템의 new_tag를 가져옴
            search_prod_tag = self.df_[self.df_['product_id'] == id_].sort_values(by='awesome_score', ascending=False)['new_tag'].values[0]
        else:
            search_prod_tag = ''
            for existing_item_id, _ in self.user_item_set:
                temp_id = self.df_[self.df_['product_id'] == existing_item_id].sort_values(by='awesome_score', ascending=False)['new_tag'].values[0]
                search_prod_tag += temp_id+' '
            search_prod_tag = search_prod_tag[:-1]
            
        # 해당 아이템-다른 아이템 간 유사도 계산
        sim_score = self.get_similarity_score(search_prod_tag, algo=algo)
        
        # save result
        sim_score = np.asarray(sim_score)  # [사용자가 선택한 아이템-다른 모든 아이템] 간 유사도 점수
        sim_score_idx = np.arange(len(sim_score)) # 유사도 점수에 대한 인덱스
        
        result_df = self.df_.iloc[sim_score_idx][['product_id', 'name', 
                                                  'new_tag', 'projectId', 
                                                  'images', 'category']]
        result_df['similarity'] = sim_score
        
        # filtering
        result_df = result_df[result_df['product_id']!=id_]  # 사용자가 선택한 아이템과 동일한 아이템 제거
        result_df = result_df.drop_duplicates(['product_id'])  # 중복된 아이템 제거
        
        # 사용자가 이미 선택한 아이템들은 추천목록에서 제거
        if len(self.user_item_set) > 1:
            print("!!사용자가 이미 선택한 아이템, 같은 카테고리 아이템들은 추천목록에서 제거!!")
            for existing_item_id, existing_item_cat in self.user_item_set:
                result_df = result_df[result_df['product_id']!=existing_item_id]
                result_df = result_df[result_df['category']!=existing_item_cat]
                
        # 유사도가 가장 높은 순서대로 정렬 => top-n개 까지 추천 결과 저장
        result_df = result_df.sort_values(by='similarity', ascending=False).reset_index()[:topn] 
        
        print(result_df)
        
        self.result = result_df
        # 추천 검색 결과 저장 및 보여주기
        self.save_result(prod, topn=topn)
        
    def save_result(self, prod, topn):
        """
        추천 검색 결과 => csv 파일, png 파일로 저장
        """
        print(f'사용자가 선택한 아이템 : {prod}')
        
        # 사용자 아이템 셋으로 부터 아이템 이름 목록 불러오기
        user_item_list = []
        for existing_item_id, _ in self.user_item_set:
                user_item_list.append(self.df_[self.df_['product_id'] == existing_item_id]['name'].values[0])
        print('사용자가 이미 선택한 아이템 목록')
        print(*user_item_list)
        print()
        
        # 검색 결과를 저장할 경로 만들기
        if not os.path.exists(f'result_{prod}'):
            os.makedirs(f'result_{prod}')
        
        # 사용자가 현재 선택한 아이템 이미지 저장
        try:
            prod_res = requests.get(self.df_[self.df_['name'] == prod]['images'].values[0][0])
            img = Image.open(BytesIO(prod_res.content))
            img.save(f'./result_{prod}/{self.user_item_len}_{prod}.png')
        except:
            print('사용자가 선택한 아이템 이미지 없음\n')

        print("="*80)
        print(f"추천 아이템 Top{topn}")
        print("="*80)
        
        # 추천된 아이템 이미지 저장
        i = 1
        for name, img_url, cat in self.result[['name', 'images', 'category']].values:
            print(f"추천 {i}순위 : {name} - {cat}")
            try:
                res = requests.get(img_url[0])
                rec_img = Image.open(BytesIO(res.content))
                rec_img.save(f'./result_{prod}/{self.user_item_len}_{name}.png')
            except:
                pass
            i += 1
        
        # 추천된 결과 DataFrame => csv 파일로 내보내기
        self.result.to_csv(f'./result_{prod}/result_{prod}.csv')
        
        
if __name__ == '__main__':
    import random
    import warnings
    
    # 경고메세지 끄기
    warnings.filterwarnings(action='ignore')
    
    base_bath = '2022-03-07/'  # 데이터 기본 경로 (필요시 변경)
    best = glob(base_bath+'best*.json')
    item = glob(base_bath+'item*.json')
    products = glob(base_bath+'products*.json')
    category = glob(base_bath+'categories.json')
    
    print("데이터를 불러옵니다.")
    st = time.time()
    DATA = Preprocess(best, item, products, category)
    print(f"데이터 전처리 시간 : {time.time()-st}초")
    
    best_item_df, products_df_4, products_df_b = DATA.best_item, DATA.products_4, DATA.products_b
    
    # qval=None => 아이템과 유사한 카테고리의 결과 추천 / qval=int => 아이템과 비슷한 스타일의 다른 아이템 추천(기존 전문가 셋을 기반)
    # algo='sorense' => 더 빠른 알고리즘 / algo='ncd' => 좀 더 다양한 추천 결과를 보여주는 알고리즘
    ent = int(input("Enterprise_Id 를 선택하세요.\n[1] 421B6D0E746C4E6D\t[2] B57D4F97C0E44A11\n"))
    if ent == 1:
        search_engine = SEARCH_RECOMMEND(products_df_4, qval=None)
        item_name_list = DATA.products_4.drop_duplicates('name')['name'].tolist()
    elif ent == 2:
        search_engine = SEARCH_RECOMMEND(products_df_b, qval=None)
        item_name_list = DATA.products_b.drop_duplicates('name')['name'].tolist()
        
    while True:
        prod = input("선택한 제품명을 입력하세요.(종료는 x)")
        if prod == 'x':
            print("추천 시스템을 종료합니다.\n감사합니다!")
            break
        if not prod:
            prod = random.choice(item_name_list)
            print(f"입력이 없어 검색할 제품명 [{prod}] 을 랜덤으로 선택합니다.")
        algo_choice = input("검색 알고리즘을 선택하세요.\n[1] Sorensen (토큰화 기반 - 빠른 알고리즘)\t[2] EntropyNCD (압축 알고리즘 - 다양한 결과)\n")
        if algo_choice == '1':
            algo = 'sorensen'
        elif algo_choice == '2':
            algo = 'ncd'
        else:
            algo = 'sorensen'
            print(f'입력이 없거나 잘못되어 default 값 [{algo}] 으로 설정합니다.')
        search_engine.search_product(prod, topn=10, algo=algo)
    
    
    