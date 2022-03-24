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
        self.best_item, self.category, self.products_4, self.products_b, self.products = self.preprocess(self.best_json, 
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

    

    def preprocess(self, best, item, products, category):
        # best.json , item.json merge시키기 위해 똑같은 columns rename 후 merge
        item.rename(columns={'enterpriseId':'enterprise_id', '_id':'id'}, inplace=True)
        templates = pd.merge(best, item, on=['enterprise_id', 'projectId', 'id'])
        
        # 2 enterprise id 
        ent2, ent1 = templates['enterprise_id'].unique().tolist()
        
        # get item_id 
        templates['items'] = templates['items'].apply(self.get_productId)
        
        # templates =  best+item
        # templates['style_predictions'] = dict(style_name: style_score) -> list(style_name: style_score)
        # edit new columns style_name, style_score > 0.1 
        templates[f'top_style_{self.style_ths_name}'] = templates['style_predictions'].apply(
            lambda x: sorted(
                [(name, score) for name, score in x.items() if score > self.style_ths],key=lambda x: x[1],reverse=True)[:3])
        
        
        # list(style_name: style_score) -> list(style_name)
        templates[f'top_style_{self.style_ths_name}'] = templates[f'top_style_{self.style_ths_name}'].apply(
            lambda x: ' '.join([name for name, score in x]))
        
        # list['items'] -> items  리스트에서 빼내오는 lambda 적용
        items_stack = pd.DataFrame(templates['items'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) 
        
        # pd.merge(templates, items_stack)
        products_df = pd.merge(templates[['enterprise_id', 
                                          f'top_style_{self.style_ths_name}', 
                                          'top_style', 'top_score',
                                          'projectId', 
                                          'awesome_score']].reset_index(), 
                               items_stack.reset_index(), 
                               on='index').drop(['index'], axis=1).rename(columns = {0:'product_id'})
        
        products_df = products_df.drop_duplicates()
        prod_tags_df = products[['_id', 'tags', 'name', 'images', 'categories']]
        
        #products_df = best+item , prod_tags_df = self.products
        products_df = pd.merge(products_df, 
                               prod_tags_df, 
                               left_on='product_id', 
                               right_on='_id').drop(['_id'], axis=1)
        
        products_df = pd.merge(products_df, 
                               products_df['product_id'].value_counts().reset_index(),
                               # values_counts() 아이템이 몇개 사용 되었는지 확인하기 위함
                               left_on='product_id',
                               right_on='index').rename(columns = {'product_id_x': 'product_id',
                                                                   'product_id_y':'use_count'}).drop(['index'], axis=1)
        
        # products_df['categories'] = list(categories) -> categories로 빼오는 과정에서 
        # 기존 stack으로 진행 했으나 일치하지 않는 문제 발생
        # code = apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True)
        
        # 현재 .apply(lambda x: x[0] if len(x) > 0 else np.nan) 진행 시 일치
        products_df['category_name'] = products_df['categories'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
        
    
        # find_category_id_preprocess
        cat_df = self.category_json
        total_cat_ids = []
        total_cat_names = []

        for i in range(len(cat_df)):
            cat_ids = []
            cat_names = []
            for child in cat_df['children'][i]:
                #category['children'][_id] 값을 찾기위함
                cat_ids.append(child['_id'])
                cat_names.append(child['name'])

                for grand_child in child['children']:
                    # category['children']['children'][_id] 값을 찾기위함
                    cat_ids.append(grand_child['_id'])
                    cat_names.append(child['name'])

            total_cat_ids.append(cat_ids)
            total_cat_names.append(cat_names)

        cat_df['cat_ids'] = total_cat_ids
        cat_df['cat_names'] = total_cat_names

        cat_ids_stack = pd.DataFrame(cat_df['cat_ids'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) # 리스트로 찾아온 값을 리스트에서 빼오는 lambda 적용
        cat_names_stack = pd.DataFrame(cat_df['cat_names'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) # 리스트로 찾아온 값을 리스트에서 빼오는 lambda 적용
        cat_ids_names_stack = pd.concat((cat_ids_stack, cat_names_stack), axis=1)

        # cat_ids_stack = cat_df['cate_ids'] 리스트 된 값들을 빼내온 DataFrame
        cat_df = pd.merge(cat_df[['name']].reset_index(), 
                   cat_ids_names_stack.reset_index(), 
                   on='index', how='left').drop(['index'], axis=1)

        cat_df.columns = ['name', 'cat_ids', 'cat_names']
        
        self.products_json['category_name'] = self.products_json['categories'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
        # products_json에서 categories list값들 빼와서 category_name 으로 지정
        prod_cat_df = pd.merge(self.products_json[['category_name',
                                                   'name',
                                                   '_id',
                                                   'tags',
                                                   'enterpriseId']],
                               cat_df,left_on = 'category_name',
                               right_on = 'cat_ids',
                               how = 'left').rename(columns={'name_x':'item_name','name_y':'category','_id':'product_id'}).drop(['category_name', 'cat_ids'], axis=1)
        
        prod_cat_df = pd.concat((prod_cat_df[prod_cat_df['enterpriseId']=='421B6D0E746C4E6D'],
                        prod_cat_df[prod_cat_df['enterpriseId']=='B57D4F97C0E44A11']))
        #products_json 에서 13개의 enterprise_id 있기에 원하는 2개의 enterpriseId 값만 추출해서 concat
        
        #사용할 컬럼만 추출해서 merge
        category_df = pd.merge(self.products_json[['_id', 'name', 'tags','color' ,'dimensions', 'images']], 
                               prod_cat_df[['item_name', 'product_id', 'enterpriseId', 'category','cat_names']], 
                                left_on = '_id',
                                right_on = 'product_id').drop(['product_id'], axis=1).rename(columns={'_id':'product_id'}).reset_index(drop=True)
        #사용할 컬럼만 추출해서 merge
        products_df = pd.merge(products_df[['projectId', 'top_style_1', 'top_style', 'top_score',
                                            'awesome_score', 'product_id', 'use_count']], 
                               category_df, 
                               how='outer').reset_index(drop=True)
        
        # products_df['projectId'] = products_df['projectId'].apply(lambda x: x.lower())
        
        products_df_4 = products_df[products_df['enterpriseId'] == ent1].reset_index(drop=True)
        #Delete useless category 
        products_df_4_new = products_df_4[pd.isnull(products_df_4['projectId'])]
        
        delete_category = ['Construction', 'Appliances', 'Bathroom', 'Kitchen', 'Outdoor']
        for i in delete_category:    
            index = products_df_4[products_df_4['category'] == i].index
            products_df_4 = products_df_4.drop(index=index).reset_index(drop=True)
            
            new_index = products_df_4_new[products_df_4_new['category'] == i].index
            products_df_4_new = products_df_4_new.drop(index=new_index).reset_index(drop=True)
        
        item_count_in_project = products_df_4.groupby(['projectId'])['product_id'].count().reset_index().rename(columns={'product_id':'item_count_in_project'})
        
        products_df_4 = pd.merge(products_df_4, item_count_in_project, on='projectId')
        products_df_4 = products_df_4[products_df_4['item_count_in_project'] > 2].reset_index(drop=True)
        
        products_df_b = products_df[products_df['enterpriseId'] == ent2].reset_index(drop=True)
        
        products_df_b_new = products_df_b[pd.isnull(products_df_b['projectId'])]
        #Delete useless category 
        delete_category = ['문/창문', '가전', '주방싱크/욕실', '파티션/구조물']
        for i in delete_category:    
            index = products_df_b[products_df_b['category'] == i].index
            products_df_b = products_df_b.drop(index=index).reset_index(drop=True)
            
            new_index = products_df_b_new[products_df_b_new['category'] == i].index
            products_df_b_new = products_df_b_new.drop(index=new_index).reset_index(drop=True)
        
        item_count_in_project = products_df_b.groupby(['projectId'])['product_id'].count().reset_index().rename(columns={'product_id':'item_count_in_project'})
        
        products_df_b_new = products_df_b[pd.isnull(products_df_b['projectId'])]

        
        products_df_b = pd.merge(products_df_b, item_count_in_project, on='projectId')
        products_df_b = products_df_b[products_df_b['item_count_in_project'] > 2].reset_index(drop=True)

        products_df_4 = pd.concat([products_df_4, products_df_4_new])
        products_df_b = pd.concat([products_df_b, products_df_b_new])
        # templates['projectId'] = templates['projectId'].apply(lambda x: x.lower())
        
        return templates, category_df, products_df_4, products_df_b ,products_df