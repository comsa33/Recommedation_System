import pandas as pd
import numpy as np
import json

class Preprocess:
    
    def __init__(self, best, item, products, category, style_ths=0.1):
        """
        style_ths : top_style_predictions 에서 범위 조절 (e.g. 0.1 이상으로 예측 점수를 받은 스타일만 받아옴 - maximum 3개)
        """
        
        self.style_ths = style_ths
        self.style_ths_name = str(style_ths)[-1]
        
        self.best = best
        self.item = item
        self.products = products
        self.category = category
        
        self.best_json = self.read_json(best).reset_index(drop=True)
        self.item_json = self.read_json(item).reset_index(drop=True)
        self.products_json = self.read_json(products).reset_index(drop=True) # products json duplicated deleted
        self.category_json = self.read_json2(category).reset_index(drop=True)
        self.best_item, self.category, self.products_4, self.products_b = self.preprocess(self.best_json, 
                                                                                                       self.item_json, 
                                                                                                       self.products_json, 
                                                                                                       self.category_json)

    def read_json(self,json_file):
        df = pd.DataFrame()
        # products.json 같은 경우 best, item.json과는 다르게 transpose로 되어있음
        if json_file != self.products:  
            for file in json_file:
                x = pd.read_json(file)
                df = pd.concat([df, x])
        else:
            # why -> products json Transpose
            for file in json_file:
                x = pd.read_json(file).T.reset_index(drop=True)    
                df = pd.concat([df, x])
        return df
    
    # category1, 2 깉은 경우 같은 값을 가지기에 1개만 적용
    # 2개의 라벨을 가지고 있어서 for 문으로 json.load
    def read_json2(self,category):
        for i in category:
            with open(i) as js:
                json_data = json.load(js)
        cat_4, cat_b = pd.DataFrame(json_data['421B6D0E746C4E6D']), pd.DataFrame(json_data['B57D4F97C0E44A11'])
        category = pd.concat([cat_4, cat_b])
        return category
    
    # item.json에서 item[items] dict(product_id)값을 꺼내오는 함수 
    def get_productId(self, items_list):
        new_list = []
        for item in items_list:
            new_list.append(item['productId'])
        return new_list
    
    ###### best_json, item_json 2개를 merge 하면 templates 
    ###### templates에서 items가 리스트로 들어있어서 lambda 적용을 해서 값만 빼오기
    def best_item_merge(self, best, item):
        item.rename(columns={'enterpriseId':'enterprise_id', '_id':'id'}, inplace=True)
        best_item = pd.merge(best, item, on=['enterprise_id', 'projectId', 'id'])

        # get item_id 
        best_item['items'] = best_item['items'].apply(self.get_productId)

        # templates =  best+item
        # templates['style_predictions'] = dict(style_name: style_score) -> list(style_name: style_score)
        # edit new columns style_name, style_score > 0.1 
        best_item[f'top_style_{self.style_ths_name}'] = best_item['style_predictions'].apply(
            lambda x: sorted(
                [(name, score) for name, score in x.items() if score > self.style_ths],key=lambda x: x[1],reverse=True)[:3])

        # list(style_name: style_score) -> list(style_name)
        best_item[f'top_style_{self.style_ths_name}'] = best_item[f'top_style_{self.style_ths_name}'].apply(lambda x: ' '.join([name for name, score in x]))


        # list['items'] -> items  리스트에서 빼내오는 lambda 적용
        items_stack = pd.DataFrame(best_item['items'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) 

        return best_item, items_stack
    
    ##### 불필요한 category 제거하는 함수
    def delete_category(self, products_df, delete_category):
        # projectId 값을 가지지 않은 데이터들 만 추출 (새로운 아이템)
        products_df_new = products_df[pd.isnull(products_df['projectId'])]

        for i in delete_category:    
            index = products_df[products_df['category'] == i].index
            products_df = products_df.drop(index=index).reset_index(drop=True)
            ###3/24 수정 
            # projectId 값을 가지지 않은 데이터들 중 카테고리 삭제 
            # 데이터 안정성을 위해 따로 진행
            new_index = products_df_new[products_df_new['category'] == i].index
            products_df_new = products_df_new.drop(index=new_index).reset_index(drop=True)


        item_count_in_project = products_df.groupby(['projectId'])['product_id'].count().reset_index().rename(columns={'product_id':'item_count_in_project'})

        products_df = pd.merge(products_df, item_count_in_project, on='projectId')
        # 프로젝트 아이디 별로 가지고있는 아이템 개수가 3개 미만 일 때 삭제 
        products_df = products_df[products_df['item_count_in_project'] > 2].reset_index(drop=True) 
        # 프로젝트 아이디 없는 아이템 과 프로젝트 아이디가 있는 아이템 결합
        # => 프로젝트 아이디가 없는 제품들은 삭제 되기에 따로 빼서 결합
        products_df = pd.concat([products_df, products_df_new])

        return products_df
     
    ##### product_json 과  category_preprocess를 거친 category와 merge 
    def products_merge_category(self, products, category_name):

        products['category_name'] = products['categories'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
                # products_json에서 categories list값들 빼와서 category_name 으로 지정
        products_merge_category = pd.merge(products[['category_name',
                                                   'name',
                                                   '_id',
                                                   'tags',
                                                   'color',
                                                   'images',
                                                   'enterpriseId']],
                               category_name,
                               left_on = 'category_name',
                               right_on = 'cat_ids',
                               how = 'left').rename(columns={'name_x':'name','name_y':'category','_id':'product_id'}).drop(['category_name', 'cat_ids'], axis=1)
        return products_merge_category
    
    ##### category_json 전처리 
    def category_preprocess(self, category):
        total_cat_ids = []
        total_cat_names = []

        for i in range(len(category)):
            cat_ids = []
            cat_names = []
            for child in category['children'][i]:
                #category['children'][_id] 값을 찾기위함
                cat_ids.append(child['_id'])
                cat_names.append(child['name'])

                for grand_child in child['children']:
                    # category['children']['children'][_id] 값을 찾기위함
                    cat_ids.append(grand_child['_id'])
                    cat_names.append(child['name'])

            total_cat_ids.append(cat_ids)
            total_cat_names.append(cat_names)

        category['cat_ids'] = total_cat_ids
        category['cat_names'] = total_cat_names

        cat_ids_stack = pd.DataFrame(category['cat_ids'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) # 리스트로 찾아온 값을 리스트에서 빼오는 lambda 적용
        cat_names_stack = pd.DataFrame(category['cat_names'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) # 리스트로 찾아온 값을 리스트에서 빼오는 lambda 적용
        cat_ids_names_stack = pd.concat((cat_ids_stack, cat_names_stack), axis=1)

        # cat_ids_stack = cat_df['cate_ids'] 리스트 된 값들을 빼내온 DataFrame
        category_name = pd.merge(category[['name']].reset_index(), 
                   cat_ids_names_stack.reset_index(), 
                   on='index', how='left').drop(['index'], axis=1)

        category_name.columns = ['name', 'cat_ids', 'cat_names']
        return category_name
    
    ###### templates 와 item_stack merge
    def best_item_merge_items_stack(self, best_item_merge, items_stack):
        
        best_item_item_stack_merge = pd.merge(best_item_merge[['enterprise_id', 
                                                              f'top_style_{self.style_ths_name}', 
                                                              'top_style',
                                                              'top_score',
                                                              'projectId', 
                                                              'awesome_score']].reset_index(), 
                                                               items_stack.reset_index(), on='index').drop(['index'], axis=1).rename(columns = {0:'product_id'})
        
        ### 중복값 제거 
        best_item_item_stack_merge_del_duplicates = best_item_item_stack_merge.drop_duplicates()
        
        return best_item_item_stack_merge_del_duplicates
        
    def preprocess(self, best, item, products, category):
        # best.json , item.json merge시키기 위해 똑같은 columns rename 후 merge
        best_item_merge, items_stack = self.best_item_merge(best, item)
        
        # pd.merge(templates, items_stack)
        best_item_item_stack_merge = self.best_item_merge_items_stack(best_item_merge, items_stack).drop_duplicates()
        
        # category_json 전처리 
        # category [['name', 'cat_ids', 'cat_names']] 만 출력
        category_name = self.category_preprocess(category)
        
        ##### product_json 과  category_preprocess를 거친 cat_df와 merge 
        products_merge_category = self.products_merge_category(products, category_name)
        
        #사용할 컬럼만 추출해서 merge
        best_item_item_stack_products_merge_products_category = pd.merge(best_item_item_stack_merge[['projectId', 'top_style_1', 'top_style', 'top_score',
                                                                                                     'awesome_score', 'product_id']], 
                                                                        products_merge_category,
                                                                        on='product_id', how='outer').reset_index(drop=True)

        #products_json 에서 13개의 enterprise_id 있기에 원하는 2개의 enterpriseId 값만 추출해서 concat
        products_df_4 = best_item_item_stack_products_merge_products_category[best_item_item_stack_products_merge_products_category['enterpriseId'] == '421B6D0E746C4E6D'].reset_index(drop=True)
        products_df_b = best_item_item_stack_products_merge_products_category[best_item_item_stack_products_merge_products_category['enterpriseId'] == 'B57D4F97C0E44A11'].reset_index(drop=True)
        
        delete_category_4 = ['Construction', 'Appliances', 'Bathroom', 'Kitchen', 'Outdoor']
        delete_category_b = ['문/창문', '가전', '주방싱크/욕실', '파티션/구조물']

        products_df_4 = self.delete_category(products_df_4, delete_category_4)
        products_df_b = self.delete_category(products_df_b, delete_category_b)
        
        return products_merge_category, category_name, products_df_4, products_df_b 