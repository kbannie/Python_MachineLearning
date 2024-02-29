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

# ## 5. 칼럼 데이터 세트 생성, 삭제
# - 생성: ____['컬럼명']=0
# - 삭제: ____ = ____.drop('컬럼명', axis=1) #inplace=False가 default값

#생성
titanic_df['Age_0']=0
titanic_df.head()

#수정
titanic_df['Agd_10']=titanic_df['Age_0']+10
titanic_df.head()

#삭제
titanic_df=titanic_df.drop('Agd_10', axis=1, inplace=False)
titanic_df.head()

# ## 6. Index 객체
# - indexes=____.index :index 객체 추출
# - indexes_new=_____.reset_index() #drop=False, inplace=False가 default값

new_value_counts=titanic_df['Pclass'].value_counts().reset_index()
new_value_counts

new_value_counts.rename(columns={'index':'Pclass', 'Plass':'Pclass_count'})

# ## 7. 인덱싱 및 필터링
# - .loc[ ] -> 명칭 기반 인덱싱
# - .iloc[ ] -> 위치 기반 인덱싱

series=titanic_df['Name']
type(series)

df=titanic_df[['Name','Age']]
type(df)

# titanic_df[0] -> 오류 발생
titanic_df[0:2]

#iloc
data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df.iloc[0,0] #슬라이싱 가능 / 불린 불가능

#loc
data_df.loc['one', 'Name'] #슬라이싱 가능 / 불린 가능

# ## 8. 정렬, Aggregation, Group by
# - 정렬 : .sort_values(by=['컬럼명']) #ascending=True가 default 값
# - Aggregation : sum(), max(), min(), count(), mean()
# - group by()

# ## 9. 기타
# - .isna() : 결손데이터
# - .fillna() : 데이터 채우기
# - .nunique() : 고유값 파악
# - .replace('특정값') : 특정갑으로 대체
