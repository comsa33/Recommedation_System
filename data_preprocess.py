import json
import pandas as pd

def read_jsonfile(json_file):
    with open(json_file) as json_file:
        json_data = json.load(json_file)
    return json_data

def get_productId(items_list):
    new_list = []
    for item in items_list:
        new_list.append(item['productId'])
    return new_list

def preprocessing(file_paths):
    
    json_list = []
    for path in file_paths:
        json_list.append(read_jsonfile(path))
    
    category_4, category_b = json_list[2]['421B6D0E746C4E6D'], json_list[2]['B57D4F97C0E44A11']
    
    bestshots_df = pd.read_json(file_paths[0])
    itemsets_df = pd.read_json(file_paths[1])
    category_en1_df = pd.DataFrame(category_4)
    category_en2_df = pd.DataFrame(category_b)
    
    itemsets_df.rename(columns={'enterpriseId':'enterprise_id', '_id':'id'}, inplace=True)
    templates_df = pd.merge(bestshots_df, itemsets_df, on=['enterprise_id', 'projectId', 'id'])
    
    ent2, ent1 = templates_df['enterprise_id'].unique().tolist()
    
    templates_df['items'] = templates_df['items'].apply(get_productId)
    
    templates_df['top3_style'] = templates_df['style_predictions'].apply(lambda x: sorted([(name, score) for name, score in x.items()], key=lambda x: x[1], reverse=True)[:3])
    templates_df['top3_style'] = templates_df['top3_style'].apply(lambda x: [name for name, score in x])
    items_stack = pd.DataFrame(templates_df['items'].apply(lambda x: pd.Series(x)).stack()).reset_index(1, drop=True) 
    products_df = pd.merge(templates_df[['enterprise_id', 'top3_style', 'top_style', 'projectId', 'awesome_score']].reset_index(), items_stack.reset_index(), on='index').drop(['index'], axis=1).rename(columns = {0:'product_id'})
    prod_tags_df = pd.DataFrame(json_list[3]).T.reset_index(drop=True)[['_id', 'tags', 'name', 'images']]
    products_df = pd.merge(products_df, prod_tags_df, left_on='product_id', right_on='_id').drop(['_id'], axis=1)
    products_df = pd.merge(products_df, products_df['product_id'].value_counts().reset_index(), left_on='product_id', right_on='index').rename(columns = {'product_id_x': 'product_id','product_id_y':'use_count'}).drop(['index'], axis=1)
    
    products_df_4 = products_df[products_df['enterprise_id'] == ent1].reset_index(drop=True)
    products_df_b = products_df[products_df['enterprise_id'] == ent2].reset_index(drop=True)
    
    return products_df_4, products_df_b