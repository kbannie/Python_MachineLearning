# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Pandas
# ## 기본
# - read_csv()
# - head()
# - tail(3)

import pandas as pd
titanic_df=pd.read_csv('titanic_train.csv')
titanic_df

titanic_df.head()

titanic_df.tail(2)

titanic_df.shape

# ## 2. DataFrame 생성

# +
dic1 = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }

# 딕셔너리를 DataFrame으로 변환
data_df=pd.DataFrame(dic1)
data_df
# -

#새로운 컬럼명 추가
#columns 인자에는 무조건 리스트만 할당 가능
data_df=pd.DataFrame(dic1, columns=['Name','Year','Gender','Age'])
data_df

#인덱스를 새로운 값으로 할당
data_df=pd.DataFrame(dic1, index=['one','two','three','four'])
data_df

# ## 3. 정보 확인
# - columns
# - index
# - index.values 
# - info() : null값 확인
# - describe()
# - value_counts(dropna=False) : 동일한 개별 데이터 값이 몇건이 있는지 확인

titanic_df.columns

titanic_df.index

titanic_df.index.values

titanic_df.info()

titanic_df.describe()

titanic_df['Pclass'].value_counts()

type(titanic_df['Pclass'])

titanic_df['Embarked'].value_counts(dropna=False)

titanic_df[['Pclass','Embarked']].value_counts(dropna=False)

type(titanic_df[['Pclass','Embarked']])

# ## 4. DataFrame과 상호변환
# - pd.DataFrame( ___, columns=['컬럼명']) : ndarray, list -> DataFrame
# - ____.values : DataFrame -> ndarray
# - ____.values.tolist() : DataFrame -> list
# - ____.values.todict() : DataFrame -> dictionary

# +
import numpy as np
#list -> DataFrame
col=['col1']
list1=[1,2,3]
arr=np.array(list1)

df_list1=pd.DataFrame(list1, columns=col)
df_list1

# +
#3개의 컬럼명 
#ndarray -> DataFrame
col=['col1','col2','col3']
list2=[[1,2,3],[4,5,6]]
arr2=np.array(list2)

df_arr2=pd.DataFrame(arr2, columns=col)
df_arr2
# -

#DataFrame -> ndarray
arr3=df_arr2.values
type(arr3)

#DataFrame -> list
list3=df_arr2.values.tolist()
type(list3)

# ## 5. 칼럼 데이터 세트 생성, 수정, 삭제
