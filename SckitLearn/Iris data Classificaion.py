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

# ## 붓꽃 품종 예측하기
# ### 1. 데이터 세트 분리
# Train data & Test data
#
# ### 2. 모델 학습
# DecisionTreeClassifier 모델을 사용하여 Train data에 fit 수행
#
# ### 3. 예측 수행
# Test data에 predict 수행
#
# ### 4. 평가
# Test data의 target 값에 accuracy_score 수행

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.metrics import accuracy_score

# +
#iris 데이터 가져오기
iris=load_iris()
iris_data=iris.data
iris_label=iris.target

#DataFrame으로 변환
iris_df=pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['target']=iris_label
iris_df.head(3)
# -

#Train/Test 분리
X_train, X_test, y_train, y_test=train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

#model 생성
model=DecisionTreeClassifier(random_state=11)
model.fit(X_train, y_train)

#예측 수행
pred=model.predict(X_test)
pred

accuracy_score(y_test, pred)
