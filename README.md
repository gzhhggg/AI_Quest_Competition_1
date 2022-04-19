# AI_Quest_Competition_1

タイトル：「PBL_02 不良個所自動検出による検品作業効率化（製造業）AI課題」  

ストーリー  
ABC基板株式会社では基板の不良個所の検品作業に工数がかかっている  
データセットを用いて基盤の合否判定が出来るAIモデルを構築・分析して結果を提出する。

課題内容    
<img src="https://user-images.githubusercontent.com/93046615/163923287-c95cd307-7cf3-47b2-ae1a-30d89eb6b6ad.png" width="900px">  

# ファイル名    

### 前処理    
1. Random_prepro.ipynb  
2. Trim_Mask_prepro.ipynb  
3. Erasing_prepro.ipynb  
4. split_img.ipynb  

### 学習  
1. learning.ipynb



# CODE  

ライブラリ  
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
import datetime
%matplotlib inline
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , accuracy_score, recall_score, precision_score, f1_score
```
前処理（Train)
