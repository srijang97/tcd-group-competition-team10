
# coding: utf-8

# In[ ]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

data_dir = ""
for dirname, _, filenames in os.walk('../../data/'):
    data_dir = dirname
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# # Import Raw Data

# In[ ]:

raw_train_df = pd.read_csv(data_dir+"/tcdml1920-rec-click-pred--training.csv")
raw_test_df = pd.read_csv(data_dir+"/tcdml1920-rec-click-pred--test.csv")


# # Basic Data Cleaning: Data split and column selection

# In[ ]:

def data_cleaning(train_df, test_df, split_method='time_viewed'):
    
    time_cols = ['query_word_count', 'query_detected_language', 'query_document_id', 'item_type', 'application_type', 
            'abstract_char_count', 'abstract_detected_language', 
            'app_lang', 'country_by_ip', 'local_hour_of_request', 'hour_request_received',
            'algorithm_class','recommendation_algorithm_id_used', 'cbf_parser', 'search_title', 'search_keywords',
            'search_abstract']    
    
   
    jabref_cols = ['query_char_count', 'app_lang', 'app_version', 'country_by_ip', 'timezone_by_ip', 
               'local_hour_of_request', 'algorithm_class', 'recommendation_algorithm_id_used']

    cbf_cols = ['cbf_parser']

    com_blog_cols = ['query_word_count', 'query_detected_language', 'abstract_char_count',
                     'application_type', 'hour_request_received', 'item_type',
                     'country_by_ip', 'algorithm_class', 'recommendation_algorithm_id_used']
    
    id_col = ['recommendation_set_id']
    
    dep_cols = ['clicks', 'ctr', 'set_clicked']
    
    train_df = train_df[train_df['rec_processing_time']<40]
    train_df = train_df[train_df['clicks']<= train_df['number_of_recs_in_set']]
    
    train_df = train_df[~(train_df['app_version'].isna())]
    
    cleaned_data = {}
    
    for key, df in {'test':test_df, 'train':train_df}.items():
        print(key)
        df = df.replace("\\N", np.nan)
        cols_to_change = ['query_word_count', 'query_char_count', 'local_hour_of_request', 'recommendation_algorithm_id_used',
                     'abstract_char_count', 'abstract_word_count']
        df[cols_to_change] = df[cols_to_change].astype('float64')
        df['q_doc_id_provided'] = df['query_document_id'].apply(lambda x: 0 if pd.isnull(x) else 1)
        
        data_dict = {}
        if split_method=='app-cbf':
            
            if key=='test':
                com_blog_cols = com_blog_cols + id_col
                jabref_cols = jabref_cols + id_col
            else:
                com_blog_cols = list(set(com_blog_cols + ['set_clicked']) - set(id_col))
                jabref_cols = list(set(jabref_cols + ['set_clicked']) - set(id_col))
                
            cblog_all = df[(df['application_type']=='blog') | (df['application_type']=='e-commerce') | (df['application_type']=='0')][com_blog_cols+cbf_cols]
        
            cblog_cbf = cblog_all[cblog_all['algorithm_class']=='content_based_filtering']
            #cblog_cbf = cblog_cbf[~(cblog_cbf['country_by_ip'].isna())]
            cblog_other = df[(df['application_type']=='blog') | (df['application_type']=='e-commerce') | (df['application_type']=='0')][com_blog_cols]
            cblog_other = cblog_other[cblog_other['algorithm_class']!='content_based_filtering']
#             cblog_other = cblog_other[~(cblog_other['country_by_ip'].isna())]
            
            dig_all = df[df['application_type']=='digital_library']
            dig_cbf = dig_all[dig_all['algorithm_class']=='content_based_filtering'][jabref_cols+cbf_cols+ ['query_detected_language', 'q_doc_id_provided']]
            #dig_cbf = dig_cbf[~((dig_cbf['country_by_ip'].isna()) | (dig_cbf['app_lang'].isna()) | (dig_cbf['app_version'].isna()))]
            dig_other = df[df['application_type']=='digital_library'][jabref_cols]
            dig_other = dig_other[dig_other['algorithm_class']!='content_based_filtering']
            #dig_other = dig_other[~((dig_other['app_lang'].isna()) | (dig_other['app_version'].isna()))]
            data_dict['cblog_cbf'] = cblog_cbf
            data_dict['cblog_other'] = cblog_other
            data_dict['dig_cbf'] = dig_cbf
            data_dict['dig_other'] = dig_other
            cleaned_data[key] = data_dict
            
        elif split_method=='time_viewed':
            if key=='test':
                df = df[~(df['application_type'].isna())]
                cleaned_data[key] = df
            else:
                df = df[(~df['time_recs_viewed'].isna()) | (df['set_clicked']==1)]
                df = df[time_cols + ['set_clicked']]
                cleaned_data[key] = df
        elif split_method=='algo':
            pass        
        
    return cleaned_data


# In[ ]:

cat_cols = ['item_type', 'application_type', 'app_lang', 'app_version', 'country_by_ip', 'timezone_by_ip', 
            'algorithm_class','recommendation_algorithm_id_used',
            'cbf_parser', 'query_detected_language', 'abstract_detected_language']

cleaned_data = data_cleaning(raw_train_df, raw_test_df, 'time_viewed')


# # Missing Value Imputation

# ## Function to get column-wise modes for missing value imputation

# In[ ]:

def group_cols(df, group, grouped):
    group_df = df.groupby(group).agg({grouped: lambda x: x.value_counts().index[0] if len(x.value_counts().index) > 0 else 'unknown'}).rename(columns={'<lambda>': grouped}).reset_index().set_index(group)
    return group_df


# ## Imputing values for all datasets 

# In[ ]:

for dataset, data in cleaned_data.items():
    for key, df in data.items():
        
        if key=='cblog_viewed':
            group_df = group_cols(cleaned_data[dataset][key], ['organization_id'], 'item_type')
            cleaned_data[dataset][key]['item_type'] = cleaned_data[dataset][key].apply(lambda x: group_df.loc[x['organization_id']][0] if pd.isnull(x['item_type']) else x['item_type'], axis=1)
    
            group_df = group_cols(cleaned_data[dataset][key], ['organization_id', 'hour_request_received'], 'country_by_ip')
            cleaned_data[dataset][key]['country_by_ip'] = cleaned_data[dataset][key].apply(lambda x: group_df.loc[(x['organization_id'], x['hour_request_received'])][0] if pd.isnull(x['country_by_ip']) else x['country_by_ip'], axis=1)
            
            group_df = group_cols(cleaned_data[dataset][key], ['organization_id'], 'cbf_parser')
            cleaned_data[dataset][key]['cbf_parser'] = cleaned_data[dataset][key].apply(lambda x: group_df.loc[x['organization_id']][0] if pd.isnull(x['cbf_parser']) else x['cbf_parser'], axis=1)
            
            missing_values = {'query_detected_language': df['query_detected_language'].mode()[0], 
#                               'abstract_char_count': df['abstract_char_count'].median(), 
                              'query_word_count': df['query_word_count'].median(),
                             'app_lang': df['app_lang'].mode()[0], 'local_hour_of_request': df['local_hour_of_request'].median(),
                     'abstract_char_count': df['abstract_char_count'].median(), 'abstract_detected_language': df['abstract_detected_language'].mode()[0]}
            
            cleaned_data[dataset][key] = df.fillna(value=missing_values)
                    
        elif key=='dig_cbf':
            
            missing_values = {'app_lang': df['app_lang'].mode()[0], 'app_version': df['app_version'].mode()[0], 'country_by_ip': 'missing', 'timezone_by_ip': 'missing', 'query_detected_language': df['query_detected_language'].mode()[0], 
                         'local_hour_of_request': df['local_hour_of_request'].median()}
            cleaned_data[dataset][key]= df.fillna(value=missing_values)
            
        else:
                        
            missing_values = {'app_lang': df['app_lang'].mode()[0], 'app_version': df['app_version'].mode()[0], 'country_by_ip': 'missing', 'timezone_by_ip': 'missing', 
                'local_hour_of_request': df['local_hour_of_request'].median(), 'recommendation_algorithm_id_used': 33}
            cleaned_data[dataset][key]= df.fillna(value=missing_values)
            


# # Data Encoding and Normalization

# ## Function to calculate weighted target encoding values

# In[ ]:

def calc_smooth_mean(df, df_train, by, on, m):
    # Compute the global mean
    mean = df_train[on].mean()

    # Compute the number of values and the mean of each group
    agg = df_train.groupby(by)[on].agg(['count', 'mean'])
    
    counts = agg['count']
    means = agg['mean']
    
    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)
#     print(smooth.index)
    # Replace each value by the according smoothed mean
    return df[by].apply(lambda x: smooth.loc[str(x)] if str(x) in smooth.index else 0)


# ## Preprocessing Function

# In[ ]:

def preprocess_data(cleaned_data, cat_cols, target, scaling_method):
    
    processed_data = {}
        
    for dataset, data in cleaned_data.items():
        processed_data[dataset] =  {}
           
        for key, df in data.items():
            
            
            print('Type: ', dataset, ' Dataset: ', key, ' Shape: ', df.shape)
            df_cat_cols = list(set(cat_cols).intersection(set(df.columns)))
            
            if dataset=='train':
                df_num_cols = list(set(df.columns) - set(df_cat_cols) -set([target]))
            else:
                df_num_cols = list(set(df.columns) - set(df_cat_cols) - set(['recommendation_set_id']))
                
            if scaling_method=='standard':
                scaler = StandardScaler()
                df[df_num_cols] = scaler.fit_transform(df[df_num_cols])
                
            else:
                scaler = MinMaxScaler()
                df[df_num_cols] = scaler.fit_transform(df[df_num_cols])
            
            for col in df_cat_cols:
#                 df_x = cleaned_data['train'][key][[col, 'set_clicked']].astype({col: str}).groupby(by=[col]).mean()           
#                 df[col] = df[col].apply(lambda x: df_x.loc[str(x)][0] if str(x) in df_x.index else 0) 
                df[col] = calc_smooth_mean(df, cleaned_data['train'][key], col, 'set_clicked', 150)
            
        
            if dataset=='train': 
                X_train, X_valid, y_train, y_valid = train_test_split(df[df_num_cols+df_cat_cols].values, df[target].values, test_size=0.3)
                processed_data[dataset][key] = {'X': df[df_num_cols+df_cat_cols].values, 'y': df[target].values,
                                                'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
            else:
                processed_data[dataset][key] = {'data': df[df_num_cols + df_cat_cols].values, 'id': df['recommendation_set_id'].values}
                
    return processed_data


# In[ ]:

processed_data = preprocess_data(cleaned_data, cat_cols, 'set_clicked', 'standard')


# # Model Application and Prediction

# In[ ]:

test_preds = {}

model = RandomForestClassifier(n_estimators=100)

for key, data in processed_data['test'].items():
    X = processed_data['train'][key]['X']
    y = processed_data['train'][key]['y']
    
    df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
    df_0 = df[df.iloc[:, -1]==0]
    df_1 = df[df.iloc[:, -1]==1]
    
    #Downsample majority class data
    df_0_new = resample(df_0, replace=True, n_samples=df_1.shape[0]*12)
    
    df = pd.concat([df_0_new, df_1])
    X = df.iloc[:, :-1].values
    y = df.iloc[: , -1].values
    
    model.fit(X, y)
    y_pred= model.predict(data['data'])
    test_preds[key] = y_pred


# # Concatenating the Results and saving to disk
# 

# In[ ]:

result = np.vstack((np.hstack((processed_data['test']['cblog_viewed']['id'].reshape(-1, 1), test_preds['cblog_viewed'].reshape(-1, 1))),
                   np.hstack((processed_data['test']['dig_cbf']['id'].reshape(-1, 1), test_preds['dig_cbf'].reshape(-1, 1))),
                   np.hstack((processed_data['test']['dig_other']['id'].reshape(-1, 1), test_preds['dig_other'].reshape(-1, 1)))))
    
result = pd.DataFrame(data = result, columns=['recommendation_set_id', 'set_clicked'])
result['recommendation_set_id'] = result['recommendation_set_id'].astype(int)
result['set_clicked'] = result['set_clicked'].astype(int)
result.sort_values(by='recommendation_set_id', inplace=True)
result.to_csv('Submission-file.csv', encoding='utf-8', index=False)

