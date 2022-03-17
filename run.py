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
import re
import random

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
        """
        # make new-tag
        # top3_style, tags, name, projectId 를 하나의 string으로 만들기
        """

        proj_ids = df.groupby('product_id')['projectId'].agg(lambda x: list(set(x))).reset_index().rename(columns={'projectId':'project_ids'})
        products_df = df.sort_values(by='awesome_score', ascending=False).drop_duplicates('product_id').reset_index(drop=True)
        products_df = df.join(proj_ids.set_index('product_id'), on='product_id')
        products_df['project_ids_str'] = products_df['project_ids'].apply(lambda x: ' '.join(x))
        products_df['weighed_project_id'] = (df.projectId.apply(lambda x : x+' ') * df.top_score.apply(lambda x : int(x*10))).tolist()
        products_df['weighed_style'] = (df.top_style.apply(lambda x : x+' ') * df.top_score.apply(lambda x : int(x*10))).tolist()
        products_df['new_tag'] = list(
                            zip(
                                products_df[f'top_style_{self.style_ths_str}'].tolist(), 
                                products_df['weighed_style'].tolist(), 
                                products_df['tags'].apply(lambda x: random.sample(x, int(len(x)*0.2))).tolist(),
                                products_df['weighed_project_id'].tolist(),
                                products_df['project_ids_str'].tolist(),
                                )
                            )
        products_df['new_tag'] = products_df['new_tag'].apply(self.reduce_newtag)
        products_df['new_tag'] = products_df['new_tag'].apply(lambda x: ' '.join(x))
        
        return products_df
    
    def reduce_newtag(self, x):
        newtag = []
        for tag in x:
            if type(tag) == str:
                newtag.append(tag)
            else:
                newtag.extend(tag)
        return newtag
    
    def add_item_in_user_item_set(self, prod, verbose=True):
        """
        # 사용자 셋에 아이템 추가
        """
        prev_len = len(self.user_item_set)
        if verbose:
            print(f'사용자 님의 아이템 목록에 [{prod}]를 추가합니다.')
        id_ = self.df_[self.df_['name'] == prod]['product_id'].values[0]
        cat_ = self.df_[self.df_['name'] == prod]['category'].values[0]
        self.user_item_set.add((id_, cat_))
        self.cold_start = False
        if verbose:
            print(f'사용자 님이 현재 선택한 아이템 목록 : {self.user_item_set}')
        if len(self.user_item_set) > prev_len:
            self.user_item_len += 1
        return id_
    
    def remove_item_from_user_item_set(self, prod, verbose=True):
        """
        # 사용자 셋에서 아이템 제거
        """
        prev_len = len(self.user_item_set)
        if verbose:
            print(f'사용자 님의 아이템 목록에서 [{prod}]를 제거합니다.')
        id_ = self.df_[self.df_['name'] == prod]['product_id'].values[0]
        cat_ = self.df_[self.df_['name'] == prod]['category'].values[0]
        self.user_item_set.remove((id_, cat_))
        if verbose:
            print(f'사용자 님의 현재 남아있는 선택된 아이템 목록 : {self.user_item_set}')
        # 사용자 목록이 비게 되면 콜드스타트
        if len(self.user_item_set) > 1:
            self.cold_start = True
        if len(self.user_item_set) < prev_len:
            self.user_item_len -= 1
            
    def get_similarity_score(self, search_prod_tag, algo='sorensen', verbose=True):
        """
        # 사용자가 선택한 아이템과 데이터 내 모든 아이템들과의 유사도 계산
        
        search_prod_tag : string, 사용자가 선택한 아이템 이름으로부터 매칭된 self.df_['new_tag'].values
        algo : 'sorensen' - 토큰 기반 거리 유사도 알고리즘, 'ncd' - 문장 압축을 통한 유사도 알고리즘
        """
        sim_score = []
        new_tag = self.df_['new_tag'].tolist()
        start_time = time.time()
        for i, tag in enumerate(new_tag):
            if algo == 'sorensen':
                sim_score.append(textdistance.Sorensen(qval=self.qval, as_set=False).normalized_similarity(search_prod_tag, tag))
            elif algo == 'ncd':
                sim_score.append(textdistance.EntropyNCD(qval=self.qval).normalized_similarity(search_prod_tag, tag))
        if verbose:
            print(f'검색 태그 : {search_prod_tag}')
            print(f'검색 시간 : {round(time.time()-start_time, 4)}초\t검색 알고리즘 : {algo}')
            print()
        return sim_score
    
    def search_product(self, 
                       prod, 
                       topn=10, 
                       algo='sorensen', 
                       from_product_id='True', 
                       base_path=None,
                       save_image=True,
                       verbose=True):
        """
        # 사용자 선택한 아이템/아이템 목록으로부터 유사도를 통해 아이템 추천
        
        prod : string, 아이템 이름
        topn : int, 유사도 상위 n개의 아이템 추천
        algo : 'sorensen' - 토큰 기반 거리 유사도 알고리즘, 'ncd' - 문장 압축을 통한 유사도 알고리즘
        from_product_id : boolean, 제품 아이디 기반으로 검색할지에 대한 파라미터
        base_path : string, 추천 결과를 저장할 기본 경로 설정
        """
        
        if from_product_id:
            # 아이템을 사용자 아이템 목록에 추가
            # declare id, category of searching product
            id_ = self.add_item_in_user_item_set(prod, verbose=verbose)
            prod_cat = self.df_[self.df_['product_id'] == id_]['category'].values[0]

            # 사용자가 현재 선택한 아이템이 속해 있는 project_id 가져오기
            self.project_id = self.df_[self.df_['name'] == prod]['project_ids'].values[0]

            if verbose:
                print(f"검색 아이템이 해당한 프로젝트 ID : {self.project_id}")
        
            # 사용자의 정보가 없는 경우 => 단일 아이템에 대한 태그 기반 추천 검색
            if self.cold_start:
                # retrieve the product tag from the input product id
                # 동일 아이템이 여러 전문가에 사용될 경우 => awesome_score 가 더 높은 점수를 받은 아이템의 new_tag를 가져옴
                search_prod_tag = self.df_[self.df_['product_id'] == id_].sort_values(by='awesome_score', 
                                                                                      ascending=False)['new_tag'].values[0]
            # 사용자 아이템 목록이 있는 경우 => 사용자가 이미 배치한 기존 아이템들까지 포함한 복수의 아이템에 대한 태그 기반 추천 검색
            else:
                search_prod_tag = ''
                for existing_item_id, _ in self.user_item_set:
                    temp_id = self.df_[self.df_['product_id'] == existing_item_id].sort_values(by='awesome_score', 
                                                                                               ascending=False)['new_tag'].values[0]
                    search_prod_tag += temp_id+' '
                search_prod_tag = search_prod_tag[:-1]
        
        else:
            search_prod_tag = prod
            
        # 해당 아이템-다른 아이템 간 유사도 계산
        sim_score = self.get_similarity_score(search_prod_tag, algo=algo, verbose=verbose)
        
        # 유사도 결과를 통해 최종 추천 아이템 테이블 만들기
        sim_score = np.asarray(sim_score)  # [사용자가 선택한 아이템-다른 모든 아이템] 간 유사도 점수
        sim_score_idx = np.arange(len(sim_score)) # 유사도 점수에 대한 인덱스
        
        result_df = self.df_.iloc[sim_score_idx][['product_id', 'name', 
                                                  'new_tag', 'projectId', 'project_ids',
                                                  'images', 'category']]
        result_df['similarity'] = sim_score
        
        # filtering : 정교한 추천 결과를 위한 검색 필터링 추가
        result_df = result_df.drop_duplicates(['product_id'])  # 중복된 아이템 제거

        # 사용자가 이미 배치한 아이템 혹은 같은 카테고리의 아이템들은 추천목록에서 제거
        if len(self.user_item_set) >= 1:
            if verbose:
                print("!!사용자가 이미 선택한 아이템, 같은 카테고리 아이템들은 추천목록에서 제거!!")
            for existing_item_id, existing_item_cat in self.user_item_set:
                result_df = result_df[result_df['product_id']!=existing_item_id]
                result_df = result_df[result_df['category']!=existing_item_cat]
                
        # 유사도가 가장 높은 순서대로 정렬 => top-n개 까지 추천 결과 저장
        self.result = result_df.sort_values(by='similarity', ascending=False).reset_index()[:topn] 
        
        if save_image:
            # 추천 검색 결과 저장 및 보여주기 => csv 파일, png 파일로 저장
            self.save_result(prod, topn=topn, base_path=base_path, verbose=verbose)
        
        return id_, prod, self.project_id, self.result['name'].tolist(), self.result['product_id'].tolist(), self.result['project_ids'].tolist(), self.result['similarity'].tolist()
        
    def save_result(self, prod, topn, base_path=None, verbose=True):
        """
        추천 검색 결과 => csv 파일, png 파일로 저장
        
        prod : string, 아이템 이름
        topn : int, 유사도 상위 n개의 아이템 추천
        base_path : 결과 저장 디렉터리 기본 경로
        """
        if verbose:
            print(f'사용자가 선택한 아이템 : {prod}')
        prod_ = re.sub(r"[^ㄱ-힣0-9a-z ]", "", prod.lower())
        
        # 사용자 아이템 셋으로 부터 아이템 이름 목록 불러오기
        user_item_list = []
        for existing_item_id, _ in self.user_item_set:
                user_item_list.append(self.df_[self.df_['product_id'] == existing_item_id]['name'].values[0])
        
        if verbose:
            print('사용자가 이미 선택한 아이템 목록')
            print(*user_item_list)
            print()
        
        # 검색 결과를 저장할 경로 만들기
        result_path = f'result_{prod_}'
        if base_path:
            result_path = base_path+'/'+result_path

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # 사용자가 현재 선택한 아이템 이미지 저장
        try:
            prod_res = requests.get(self.df_[self.df_['name'] == prod]['images'].values[0][0])
            img = Image.open(BytesIO(prod_res.content))
            img.save(result_path+f'/{self.user_item_len}_{prod_}.png')
        except:
            if verbose:
                print('사용자가 선택한 아이템 이미지 없음\n')
        
        if verbose:
            print("="*80)
            print(f"추천 아이템 Top{topn}")
            print("="*80)

        # 추천된 아이템 이미지 저장
        i = 1
        for name, img_url, cat in self.result[['name', 'images', 'category']].values:
            if verbose:
                print(f"추천 {i}순위 : {name} - {cat}")
            name_ = re.sub(r"[^ㄱ-힣0-9a-z ]", "", name.lower())
            try:
                res = requests.get(img_url[0])
                rec_img = Image.open(BytesIO(res.content))
                rec_img.save(result_path+f'/{self.user_item_len}_{name_}.png')
            except:
                pass
            i += 1

        # 추천된 결과 DataFrame => csv 파일로 내보내기
        self.result.to_csv(result_path+f'/result_{prod_}.csv')
        

if __name__ == '__main__':
    import random
    import warnings
    
    # 경고메세지 끄기
    warnings.filterwarnings(action='ignore')
    
    # 데이터 경로 지정
    base_bath = '2022-03-14/'  # 데이터 기본 경로 (필요시 변경)
    best = glob(base_bath+'best*.json')
    item = glob(base_bath+'item*.json')
    products = glob(base_bath+'prod*.json')
    category = glob(base_bath+'cate*.json')
    
    # json => DataFrame 데이터 불러오기
    print("데이터를 불러옵니다.")
    st = time.time()
    DATA = Preprocess(best, item, products, category)
    print(f"데이터 전처리 시간 : {time.time()-st}초")
    
    best_item_df, products_df_4, products_df_b = DATA.best_item, DATA.products_4, DATA.products_b
    
    # 추천 프로그램 실행
    # qval=None => 아이템과 유사한 카테고리의 결과 추천 / qval=int => 아이템과 비슷한 스타일의 다른 아이템 추천(기존 전문가 셋을 기반)
    # algo='sorense' => 더 빠른 알고리즘 / algo='ncd' => 좀 더 다양한 추천 결과를 보여주는 알고리즘
    while True:
        ent = int(input("Enterprise_Id 를 선택하세요.\n[1] 421B6D0E746C4E6D\t[2] B57D4F97C0E44A11\n"))
        if ent == 1:
            search_engine = SEARCH_RECOMMEND(products_df_4, qval=3)
            item_name_list = DATA.products_4.drop_duplicates('name')['name'].tolist()
            break
        elif ent == 2:
            search_engine = SEARCH_RECOMMEND(products_df_b, qval=3)
            item_name_list = DATA.products_b.drop_duplicates('name')['name'].tolist()
            break
        else:
            print("입력된 정보가 정확하지 않습니다. 다시 입력해주세요.")
    
    from_product_id_input = input("[1] 아이템 이름으로 추천 검색\t[2] 카테고리 별 추천 검색")
    if from_product_id_input == '1':
        from_product_id = True
    elif from_product_id_input == '2':
        from_product_id = False
    else:
        from_product_id = True
        print("입력이 없어 '아이템 이름으로 추천 검색'을 합니다.'")
    
    while True:
        if from_product_id:
            prod = input("선택한 제품명을 입력하세요.\n(랜덤 선택은 입력 없이 enter, 프로그램 종료는 x)")
            
        else:
            prod = input("검색어를 입력하세요.\n(랜덤 선택은 입력 없이 enter, 프로그램 종료는 x)")
        
        if not prod:
                prod = random.choice(item_name_list)
                print(f"입력이 없어 검색할 제품명 [{prod}] 을 랜덤으로 선택합니다.")    
        if prod == 'x':
            print("추천 시스템을 종료합니다.\n감사합니다!")
            break
        
        algo_choice = input("검색 알고리즘을 선택하세요.\n[1] Sorensen (토큰화 기반 - 빠른 알고리즘)\t[2] EntropyNCD (압축 알고리즘 - 다양한 결과)\n")
        if algo_choice == '1':
            algo = 'sorensen'
        elif algo_choice == '2':
            algo = 'ncd'
        else:
            algo = 'sorensen'
            print(f'입력이 없거나 잘못되어 default 값 [{algo}] 으로 설정합니다.')
        search_engine.search_product(prod, topn=10, algo=algo, from_product_id=from_product_id)
    
    
    