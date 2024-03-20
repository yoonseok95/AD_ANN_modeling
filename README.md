**Description**

아토피 피부염 환자들의 데이터를 가지고 컴퓨터가 인식할 수 있게 한 후 학습시켜 결과적으로, 새로운 데이터(다른 사람)를 넣었을 때 컴퓨터가 아토피 피부염 질환의 유무를 분류할 수 있도록 만든 머신러닝이다. 

> 머신러닝은 학습할 자료(데이터) 처리에 인간이 선 개입해 컴퓨터가 인식할 수 있게 한 후 학습하게 해서 모델을 만들어내는 것이다.
> 아래는 머신러닝 분류모델을 만드는 방법이다
1. 데이터 수집
2. 데이터 전처리
3. 분석 모델링을 통해 모델 selection
4. 모델 training (split)
5. 모델 평가 검증


# 데이터 수집
pd.read_csv로 AD data를 불러온다. 
```
df = pd.read_csv("normalized_AD.csv)
df = df.iloc[:, 1:]
```

![image](https://github.com/kimahyun0606/AD/assets/162280996/11022350-54aa-4632-821d-cb5d013d16c2)


#### 데이터 프레임 구성
df = pd.DataFrame(df)
df.head()
![image](https://github.com/kimahyun0606/AD/assets/162280996/687303db-7cd1-48d6-a881-8eac672cfc56)

    
    
# 데이터 상관관계 분석(전처리 과정)

최초에 수집된 데이터는 다양한 이유로 분석을 방해하는 요소가 존재한다. 데이터가 목적에 맞게 최적화되어 있지 않기 때문에 수집데이터를 그대로 사용할 경우 잘못된 분석결과를 도출하거나 분석의 성능이 떨어질 수 있다. 때문에 데이터 전처리 과정은 매우 중요하게 다루어 지고 있다. 

> 이 과정을 통해 값이 누락된 데이터 결측치와 일반적인 범위에서 벗어난 값 이상치를 제외하거나 적절히 수정하여 분석의 정확성을 높인다.    

> 변수들 간의 영향력을 조정하기 위해 정규화와 표준화를 사용한다. 데이터 변수들 간의 범위가 다를 경우 분석의 성능이 하락할 수 있기 때문이다.    

> 전체 데이터 중 분석 영향력이 떨어지는 변수를 제거하여 분석의 성능을 높이는 전처리 과정인 '피처 선택'과 수집 데이터에 존재하는 변수들 간의 연산을 통해 파생 변수를 생성하는 것을 '피처 엔지니어링'과성을 통해 모델의 복잡성을 줄이고 효율성을 높이고, 모델의 예측 성능을 향상시킬수 있다.   

### 데이터 불러오기
library(readxl)   
library(ggplot2)   
library(dplyr)   
library(rlang)   

AD <- read.csv("AD.csv")   
AD <- AD[, -1]   
head(AD)   
install.packages("ltm")   
library(ltm)   
![image](https://github.com/kimahyun0606/AD/assets/162280996/b5b054e5-bd1f-459d-8abe-4bd6dde05a88)


### correlations test 반복문
```
cor_results <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```


> ### for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:30) { 
  cor_test <- cor.test(AD[, i], AD$AD)
```   
>> ### 결과를 데이터프레임에 추가
```
  cor_results <- rbind(cor_results, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
 }
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/d9f1283d-7629-444d-9904-98a8f7850358)

### 결과 데이터프레임 출력
```
print(cor_results)
cor_results

write.csv(cor_results, file ="cor_results.csv")
```

=========


### correlations test (MetRate제거)
```
AD_1 <- AD %>% select(-MetRate)
head(AD_1)

cor_results_1 <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```


> ### for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:29) { 
  cor_test <- cor.test(AD_1[, i], AD_1$AD)
```

>> ### 결과를 데이터프레임에 추가
```
  cor_results_1 <- rbind(cor_results_1, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD_1)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/6be6b23b-1300-4094-83c7-68dd5db700f4)

### 결과 데이터프레임 출력
```
print(cor_results_1)
cor_results_1

write.csv(cor_results_1, file ="cor_results_1.csv")
```
=======


t_test_results <- data.frame(Variable1 = character(), Variable2 = character(), statistic = numeric(), P_Value = numeric())

### t- test 반복문
```
AD_1 <- AD %>% filter(AD ==1)
summary(AD_1)

AD_0 <- AD %>% filter(AD==0)
summary(AD_0)
```
```
> for (i in 2:30) { 
  t_test <- t.test(AD_1[, i], AD_0[, i])
```

>> ### 결과를 데이터프레임에 추가
```
  t_test_results <- rbind(t_test_results, data.frame(
    Variable1 = "AD",
    Variable2 = "non-AD",
    variable3 = names(AD_1)[i],
    t_value = t_test$statistic,
    P_Value = t_test$p.value
  ))
}
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/d90023b5-3110-4f6c-8327-dbfc7ff48133)

```
t_test_results
write.csv(t_test_results, file="t_test_results.csv")
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/272c9bd6-d63a-46e6-94bf-c00a4f812df6)
![image](https://github.com/kimahyun0606/AD/assets/162280996/dba2473d-afca-481c-af63-3357ae36cb8c)


#R프로그램으로 돌려 나온 일련의 Data file
![image](https://github.com/kimahyun0606/AD/assets/162280996/5283da6c-bb4f-49eb-88e8-19268692769b)
![image](https://github.com/kimahyun0606/AD/assets/162280996/d846a4f8-1f68-4d87-b3d2-d6f12f8dd754)


# 분석 모델링 선택
ANN, LR, NB, RF, SVM 5개 모델을 생성하였다. 

![image](https://github.com/kimahyun0606/AD/assets/162280996/d04f82b2-f314-4a02-b063-bdd72278a108)

피처 4개 Fe, Cu, Pb, Na_Mg

### 패키지 불러오기
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### ADdata.csv불러오기
```
df = pd.read_csv("normalized_AD.csv")
df = df.iloc[:, 1:]
```
```
df = df[['AD','Fe','Cu','Pb','Na_Mg']]
```

### 데이터 프레임 구성
```
df = pd.DataFrame(df)
df.head()
```
```
df.info()
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/6058e88d-c4bb-4f2d-a54d-afdce4653ba8)

### 데이터 개수 세기
```
print(df['AD'].value_counts())
print(df['sex'].value_counts())
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/a753b429-d091-44ab-96b2-4e0b1833ab47)

### 칼럼 개수
```
print(df.columns)
print(df.shape[1])
```

### matplotlib 한글 폰트 추가하기
```
import matplotlib.font_manager as fm

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

f = [f.name for f in fm.fontManager.ttflist]
print(len(font_list))

plt.rcParams["font.family"] = 'Malgun Gothic'
```

### 음수 표시
```
plt.rcParams['axes.unicode_minus'] = False
```
```
X = df.loc[:,df.columns !='AD']
y = df['AD']
```

여기까지는 5개의 모델링 코드가 같다.  

### 데이터 split 분리

#라이브러리 패키지 불러오기

ANN
```
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
```

LR
```
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import statsmodels.api as sm
```

NB
```
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
```

RF
```
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```

SVM
```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
```

#훈련데이터 평가 데이터 분리
> X, y로 일단 split 하고, train_test_split(X, y, stratify, test_size, random_state)
```
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=5) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/439362a3-63af-461a-8e8c-72e3cf6654b0)


# 분석 모델 적용

> ANN 적용하기
>> 그리드 서치로 하이퍼 파라미터 조정(검증)
```
ann = MLPClassifier()

params = {
    'hidden_layer_sizes':[10,30,50,100],
    'activation':['relu','tanh'],
    'solver':['adam','sgd'],
    
}

grid = GridSearchCV(ann,param_grid=params,verbose=1)
grid.fit(x_train,y_train)

print(grid.best_params_)
```

```
ann = MLPClassifier(hidden_layer_sizes= 50,solver='sgd',activation='tanh')
ann.fit(x_train,y_train)
y_pred = ann.predict(x_test)
print(classification_report(y_test,y_pred))
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/eb7eb6d2-e796-4925-ae22-cc08f949abd1)


> LR 적용하기 
>> 그리드 서치로 최적의 하이퍼 파라미터 구하기

```
LR =  LogisticRegression

params = { 'penalty' : ['l2','l1'],
         'C':[0.01, 0.1, 1, 5, 10]}

grid = GridSearchCV(model, param_grid = params, verbose=1)
# grid.fit(data_scaled, cancer.target)
grid.fit(x_train, y_train)
print(grid.best_params_)
```
```
model = LogisticRegression  
model.fit(x_train, y_train)   
y_pred =model.predict(x_test)   
print(classification_report(y_test, y_pred))
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/ed4fafd8-0cca-45ea-9957-2e5fbeebacc2)


> NB 적용하기
>> nb = BernoulliNB
```
nb = BernoulliNB()

params = { 'alpha' :[1,10,50,100] }

grid = GridSearchCV(nb, param_grid=params, verbose=1)
grid.fit(x_train, y_train)
print(grid.best_params_)
```
```
nb = BernoulliNB(alpha=1)
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
print(classification_report(y_test,y_pred))
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/c570af94-a42f-479e-ab41-766c7930064d)


> RF 적용하기
>> GridSearch를 이용하여 하이퍼파라미터 조정하기(검증)
```
rf = RandomForestClassifier()

params = { 'n_estimators' : [ 20, 40, 60, 80, 100],
           'max_depth':['None',4,8,12,16],
            'criterion' :['gini','entropy'],
           'min_samples_leaf' : [0, 2, 4, 6,8,10],
           'min_samples_split' : [1, 5, 10, 15]
            }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
grid = GridSearchCV(rf, param_grid = params,verbose=1)
grid.fit(x_train, y_train)
print(grid.best_params_)
```
```
rf = RandomForestClassifier(n_estimators=80, 
                             criterion='entropy',
                             max_depth =12,
                             min_samples_leaf=6,
                             min_samples_split =15)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
print(classification_report(y_test,y_pred))
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/ba5acfe6-3951-4b08-a1c6-bfd99026d8e2)


> SVM 적용하기
>> 하이퍼 파라미터 조정 없이 진행
```
svm =SVC()

params = {
    'kernel':['linear', 'poly', 'rbf'],
    'gamma':[0.00001,0.0001,0.001,0.01,0.1,1],
    'C': [0.01,0.1,1,10,100,1000]
}

    
grid =  GridSearchCV(svm,param_grid= params,verbose=1)
grid.fit(x_train, y_train)
print(grid.best_params_)
```
```
svm =SVC(C=1, kernel = 'rbf', gamma=0.1)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

print(classification_report(y_test,y_pred))
```
![image](https://github.com/kimahyun0606/AD/assets/162280996/b36d51b2-0cfe-4e83-a7bc-e95cd2b1c883)

# 데이터 분석 결과 검증
#'함수이름'.predict_proba('X test 데이터) : roc auc score를 구할때 사용
from sklearn.metrics import roc_auc_score >>>roc_auc_score(Y test 데이터, Y 예측 데이터)
```
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
labels = [0,1]
y_test =label_binarize(y_test,classes=labels)
y_pred =label_binarize(y_pred,classes=labels)
```

```
n_classes = 1
fpr = dict()
tpr =dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_test[:,i],y_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])
```

```
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
for idx, i in enumerate(range(n_classes)):
    plt.subplot(141+idx)
    plt.plot(fpr[i],tpr[i],label='ROC curve (area = %0.2f)'%roc_auc[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.legend(loc="lower right")
plt.show()
```
결과 : 
> ANN ![image](https://github.com/kimahyun0606/AD/assets/162280996/dea92d47-1a09-4cb7-b5fb-53e852646498)    
   
> LR ![image](https://github.com/kimahyun0606/AD/assets/162280996/60089692-8fc9-4dec-9235-49cf42750652)    
    
> NB ![image](https://github.com/kimahyun0606/AD/assets/162280996/4b8b5238-e251-4b65-b920-d4f498119c99)

> RF ![image](https://github.com/kimahyun0606/AD/assets/162280996/72b9496f-7e4d-4ba3-b3f2-88538029f81d)

> SVM ![image](https://github.com/kimahyun0606/AD/assets/162280996/4340a5aa-5974-4c46-97c5-a0d110489fa6)

```
print("roc_auc_score:",roc_auc_score(y_test,y_pred, multi_class='raise'))
```
결과 : 
> ANN roc_auc_score: 0.5666666666666667    
> LR roc_auc_score: 0.7333333333333333    
> NB roc_auc_score: 0.44999999999999996    
> RF roc_auc_score: 0.7333333333333334    
> SVM roc_auc_score: 0.75   
