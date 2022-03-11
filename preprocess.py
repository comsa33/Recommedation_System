class Preprocess:
    
    def __init__(self, best, item, products, category):
        self.best = self.read_json(best)
        self.item = self.read_json(item)
        self.products = self.read_json(products).drop_duplicates('_id') # products json duplicated deleted
        self.category = self.read_json2(category).reset_index(drop=True)
        self.best_item, self.products_4, self.products_b = self.preprocess(self.best, self.item, self.products, self.category)
        
    def read_json(self,json_file):
        df = pd.DataFrame()
        if json_file != products:  
            for file in json_file:
                x = pd.read_json(file)
                df = pd.concat([df, x])
        else:
            # why -> products json duplicated 
            for file in json_file:
                x = pd.read_json(file).T.reset_index(drop=True)    
                df = pd.concat([df, x])
        return df

    def read_json2(self,category):
        for i in category:
            with open(i) as js:
                json_data = json.load(js)
        cat_4, cat_b = pd.DataFrame(json_data['421B6D0E746C4E6D']), pd.DataFrame(json_data['B57D4F97C0E44A11'])
        category = pd.concat([cat_4, cat_b])
        return category
    
    def get_productId(self, items_list):
        new_list = []
        for item in items_list:
            new_list.append(item['productId'])
        return new_list
    
    # category['children'] -> name(label) 
    def find_category_df(self, category):
        df = pd.DataFrame()
        for i in category['children']:
            df2 = pd.DataFrame(i)
            df = pd.concat([df, df2])
        return df
    
    def preprocess(self, best, item, products, category):
        
        item.rename(columns={'enterpriseId':'enterprise_id', '_id':'id'}, inplace=True)
        templates = pd.merge(best, item, on=['enterprise_id', 'projectId', 'id'])
        
        # 2 enterprise id 
        ent2, ent1 = templates['enterprise_id'].unique().tolist()
        
        # get item_id 
        templates['items'] = templates['items'].apply(self.get_productId)
        
        # find_category_id_preprocess
        cat_df = self.find_category_df(self.category) 
        cat_df2 = self.find_category_df(cat_df)
        category_df = pd.concat([cat_df, cat_df2], ignore_index=True)
        category_df = pd.merge(self.category[['name', '_id']], category_df, left_on='_id', right_on='parentId').rename(columns={'_id_y' : '_id'}).drop(columns=['_id_x'])
        
        # edit new columns style_name, style_score > 0.1 
        templates['top_style_1'] = templates['style_predictions'].apply(lambda x: sorted([(name, score) for name, score in x.items() if score > 0.1], key=lambda x: x[1], reverse=True)[:3])
        # del list, style score
        templates['top_style_1'] = templates['top_style_1'].apply(lambda x: [name for name, score in x])
        # Edit best_item['items'] = list(values) -> values
        items_stack = pd.DataFrame(templates['items'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) 
        products_df = pd.merge(templates[['enterprise_id', 'top_style_1', 'top_style', 'projectId', 'awesome_score']].reset_index(), items_stack.reset_index(), on='index').drop(['index'], axis=1).rename(columns = {0:'product_id'})
        prod_tags_df = products[['_id', 'tags', 'name', 'images', 'categories']]
        products_df = pd.merge(products_df, prod_tags_df, left_on='product_id', right_on='_id').drop(['_id'], axis=1)
        products_df = pd.merge(products_df, products_df['product_id'].value_counts().reset_index(),left_on='product_id',right_on='index').rename(columns = {'product_id_x': 'product_id','product_id_y':'use_count'}).drop(['index'], axis=1)
        
        # category[categories] = list(values) -> values
        products_df['category_name'] = products_df['categories'].apply(lambda x: pd.Series(x)).reset_index(drop=True).drop(columns=1)
        
        # category_df['name_x'] = category['children']
        # category_df['name_y'] = category['children'][children]
        products_df = pd.merge(products_df, category_df[['_id','name_x', 'name_y']], left_on='category_name', right_on='_id', how='left')
        
        products_df_4 = products_df[products_df['enterprise_id'] == ent1].reset_index(drop=True)
        products_df_b = products_df[products_df['enterprise_id'] == ent2].reset_index(drop=True)
        
        return templates, products_df_4, products_df_b
