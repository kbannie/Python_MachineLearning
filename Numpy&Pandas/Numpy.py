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

# # Numpy
# ## 1. 기본
# ### (1) ndarray로 변환
# ndarray=np.array([1,2,3],[4,5,6]) #리스트 형태의 인자를 받음
# ### (2) 차원 크기 확인
# ndarray.shape #(2,3)
# ### (3) 데이터 타입
# ndarray.dtype #int32
# ### (4) 형변환
# ndarray.astype('float64') #float64로 변환
#
# -
#
# ## 2. 초기화 및 차원크기변경
# ### (1) 순차적으로 초기화
# array1=np.arange(10) #[0 1 2 3 4 5 6 7 8 9]
# ### (2) 0으로 초기화
# np.zeros((3,2), dtype='int32') #튜플 형태의 인자를 받음
# ### (3) 1로 초기화
# np.ones((3,2)) #튜플 형태의 인자를 받음
# ### (4) 차원 크기 변경
# array2=array1.reshape(2,5) #(2,5) 크기의 2차원으로 변경 #튜플로 넣어도 가능
# array2=array1.reshape(-1,5) #열이 5이면서 행은 알아서 맞추도록 함
# array2=array1.reshape(-1,) #2차원을 1차원으로 변환
#
# -
#
#
# ## 3. 인덱싱
# 0부터 시작
# ### (1) 단일 인덱싱
# array1[1,2]
# ### (2) 슬라이시 인덱싱
# array1[0:3,1:4] #행은 0~2 / 열은 1~3
# ### (3) 팬시 인덱싱
# array1[[0,2].2] #행은 0,2 / 열은 2
# ### (4) 불린 인덱싱
# array3=array1[array1>5] 
# #조건:array1에서 5보다 큰 값들 -> True/False로 변환 -> True 인덱스에 해당하는 값들을 array3에 넣기
#
# -
#
#
# ## 4. 배열의 정렬
# ### (1) sort()
# sort_arr=np.sort(arr) #arr의 정렬 배열을 반환
# arr.sort() #arr 자체를 정렬
# sort_arr_desc=np.sort(arr)[::-1] #내림차순
# sort_arr_axis0=np.sort(arr, axis=0) #행 방향으로 정렬 / axis=1 -> 열 방향으로 정렬
# ### (2) argsort()
# argsort_arr=np.argsort(arr) #정렬된 배열의 인덱스를 반환

import numpy as np
array1=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array1, type(array1))

array1.shape

array2=np.arange(10)
array2

array0=np.zeros(10, dtype='int32')
array0

array3=array2.reshape(-1,2)
array3

array4=array3.reshape(-1)
array4

array20=np.arange(20)
array202=array20.reshape(-1,5)
array202

array202[:2, 2:]

array202[[2,3]]

array2021=array202[array202>9]
array2021

array2d = np.array([[8, 12], [7, 13 ]])
array2d_a0=np.sort(array2d, axis=0)
array2d_a0


