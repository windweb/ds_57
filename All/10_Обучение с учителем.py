#Подготовка признаков

1. Загрузите данные из файла /datasets/travel_insurance.csv в переменную data. Распечатайте первые десять элементов на экране. Изучите данные.
import pandas as pd

data = pd.read_csv('/datasets/travel_insurance.csv')
print(data.head(10))

2. Разбейте исходные данные на две выборки:
обучающую (train);
валидационную (valid). Это 25% исходных данных. Установите параметр (random_state) равным 12345. Объявите четыре переменные и запишите в них:
признаки: features_train, features_valid;
целевой признак: target_train, target_valid.
Выведите на экран размеры таблиц, которые хранятся в переменных: features_train и features_valid.

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance.csv')
features = data.drop('Claim', axis=1)
target = data['Claim']

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

print(features_train.shape)
print(features_valid.shape)

#Пробное обучение

Задача
Проверьте, признаки какого типа хранятся в таблице. Выведите их на экран. Затем напечатайте первые пять значений столбца Gender.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance.csv')

print(data.dtypes)
print(data['Gender'].head())

#Прямое кодирование
Задача
Преобразуйте колонку Gender техникой OHE. Вызовите функцию pd.get_dummies() и напечатайте на экране первые пять записей изменённой таблицы.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance.csv')

print(pd.get_dummies(data['Gender']).head())

#Дамми-ловушка

1. Преобразуйте колонку Gender техникой OHE. 
Чтобы не попасть в дамми-ловушку, примените аргумент drop_first функции pd.get_dummies(). Напечатайте первые пять записей изменённой таблицы.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance.csv')
# < напишите код здесь >

print(pd.get_dummies(data['Gender'], drop_first=True).head())

2. Примените прямое кодирование ко всему датафрейму.
Вызовите функцию pd.get_dummies() c аргументом drop_first. Сохраните таблицу в переменной data_ohe.
Выведите на экран первые три строки преобразованной таблицы.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance.csv')

print(pd.get_dummies(data, drop_first=True).head(3))

3. Разбейте исходные данные на две выборки в соотношении 75:25 (%):
    обучающую (train);
    валидационную (valid).
Объявите четыре переменные и запишите в них:
    признаки: features_train, features_valid;
    целевой признак: target_train, target_valid.
Вам предстоит освоить альтернативный способ работы с функцией train_test_split(): когда на вход подаются две переменные (признаки и целевой признак). Поработайте с документацией.
Обучите логистическую регрессию. Напечатайте на экране текст "Обучено!" (уже в прекоде). Так вы убедитесь, что код выполнился без ошибок.
Вложите и в train_test_split(), и в LogisticRegression() параметр random_state, равный 12345.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) 

model = LogisticRegression(random_state=12345,solver='liblinear')
model.fit(features_train, target_train)
print("Обучено!")
#score = model.score(features_valid, target_valid)

#Порядковое кодирование

1. Преобразуйте признаки техникой Ordinal Encoding. Импортируйте OrdinalEncoder из модуля sklearn.preprocessing.
Сохраните результат в переменной data_ordinal. Оформите данные в структуру DataFrame().
Напечатайте на экране первые пять строк таблицы (уже в прекоде).

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder 
# < напишите код здесь >

data = pd.read_csv('/datasets/travel_insurance.csv')
encoder = OrdinalEncoder() 
encoder.fit(data) 
data_ordinal = encoder.transform(data) 
data_ordinal = pd.DataFrame(encoder.transform(data), columns=data.columns) 
# < напишите код здесь >

print(data_ordinal.head())

2. Обучите решающее дерево на преобразованных данных.
Напечатайте на экране текст "Обучено!" (уже в прекоде). Так вы убедитесь, что код выполнился без ошибок.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance.csv')

encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data),
                            columns=data.columns)

target = data_ordinal['Claim']
features = data_ordinal.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train) 
print("Обучено!")


#Масштабирование признаков
Задача
Стандартизируйте численные признаки. Импортируйте StandardScaler из модуля sklearn.preprocessing.
Создайте объект структуры StandardScaler() и настройте его на обучающих данных. В переменной numeric уже есть список всех численных признаков.
Сохраните преобразованные обучающую и валидационную выборки в переменных: features_train и features_valid.
Напечатайте на экране первые пять строк таблицы.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# < напишите код здесь >

data = pd.read_csv('/datasets/travel_insurance.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])

features_valid[numeric]  = scaler.transform(features_valid[numeric])

pd.options.mode.chained_assignment = None

print(features_train.head())


#Метрики классификации

#Accuracy для решающего дерева
Задача
Обучите модель решающего дерева. Посчитайте значение accuracy на валидационной выборке. Сохраните результат в переменной accuracy_valid. Напечатайте его на экране.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train) 
predicted_valid = model.predict(features_valid)
accuracy_valid = accuracy_score(target_valid, predicted_valid) 
print(accuracy_valid)

#Проверка адекватности модели

1. Для подсчёта классов в целевом признаке примените метод value_counts(). Сделайте частоты относительными (от 0 до 1): в этом поможет документация Pandas.
Значения сохраните в переменной class_frequency. Напечатайте их на экране.
Методом plot() c аргументом kind='bar' постройте диаграмму.


import pandas as pd
data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

class_frequency = data['Claim'].value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')

2. Проанализируйте частоты классов в результатах предсказаний решающего дерева (переменная predicted_valid).
Всё то же самое:
Примените метод value_counts(). Сделайте частоты относительными.
Значения сохраните в переменной class_frequency. Напечатайте их на экране.
Методом plot() c аргументом kind='bar' постройте диаграмму.


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

# чтобы работала функция value_counts(),
# мы преобразовали результат к pd.Series 
predicted_valid = pd.Series(model.predict(features_valid))

# ... (загрузка данных и обучение) ...

class_frequency = predicted_valid.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')

3. Создайте константную модель: любому объекту она прогнозирует класс «0». Сохраните её предсказания в переменной target_pred_constant.
Напечатайте на экране значение accuracy.

import pandas as pd
from sklearn.metrics import accuracy_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)

target_pred_constant = pd.Series(0, index=target.index)
print(accuracy_score(target, target_pred_constant)) 

#Баланс и дисбаланс классов
#Истинно положительные ответы

Задача
Мы сделали пример предсказаний и правильных ответов. Посчитайте количество TP-ответов и напечатайте результат на экране.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

tp = ((target == 1) & (predictions == 1)).sum()
print(tp)

#Истинно отрицательные ответы
Посчитайте количество TN-ответов, как и в прошлой задаче. Напечатайте результат на экране.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

tn = ((target == 0) & (predictions == 0)).sum()
print(tn)

#Ложноположительные ответы

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

fp = ((target == 0) & (predictions == 1)).sum()
print(fp)

#Ложноотрицательные ответы

Посчитайте количество FN-ответов. Напечатайте результат на экране.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

fn = ((target == 1) & (predictions == 0)).sum()
print(fn)


#Матрица ошибок
1. Рассчитайте матрицу ошибок функцией confusion_matrix(). Импортируйте её из модуля sklearn.metrics. Напечатайте результат на экране.

import pandas as pd
from sklearn.metrics import confusion_matrix

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

print(confusion_matrix(target, predictions))


2. Постройте матрицу ошибок для решающего дерева. Как и в прошлом задании, вызовите функцию confusion_matrix(). 
Напечатайте результат на экране.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print(confusion_matrix(target_valid,predicted_valid))

#Полнота
Задача
Найдите в модуле sklearn.metrics функцию, которая отвечает за вычисление полноты. Импортируйте её.
Функция принимает на вход верные ответы и предсказания, а возвращает долю правильных ответов, найденных моделью. Напечатайте результат на экране.


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print(recall_score(target_valid, predicted_valid))

#Точность
Задача
Найдите в модуле sklearn.metrics функцию, которая отвечает за вычисление точности. Импортируйте её.
Функция принимает на вход верные ответы и предсказания. Возвращает, какая доля объектов, отмеченных моделью как положительные, на самом деле такие. Напечатайте результат на экране.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print(precision_score(target_valid, predicted_valid))

#Точность против полноты
#F1-мера

1. Посчитайте:
точность, применив функцию precision_score();
полноту функцией recall_score();
F1-меру по формуле из теории.
Сохраните метрики в переменных: precision, recall и f1.
Напечатайте их значения на экране (уже сделано в прекоде).

import pandas as pd
from sklearn.metrics import recall_score,precision_score

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

precision = precision_score(target, predictions)
recall = recall_score(target, predictions)
f1 = (2 * precision * recall)/(precision+recall)

print("Полнота:", recall)
print("Точность:", precision)
print("F1-мера:", f1)


2. Найдите в модуле sklearn.metrics функцию, которая отвечает за вычисление F1-меры. Импортируйте её.
Функция принимает на вход верные ответы и предсказания, а возвращает среднее гармоническое точности и полноты. Напечатайте результат на экране.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))

#Взвешивание классов
#Задача
Перед вами код обучения логистической регрессии с равнозначными классами из прошлых уроков.
Сделайте веса классов сбалансированными. Обратите внимание, как изменится значение F1-меры.

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear',class_weight='balanced' )
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print("F1:", f1_score(target_valid, predicted_valid))

#Увеличение выборки
#Задача
1.
Мы разделили обучающую выборку на отрицательные и положительные объекты.
Объявите четыре переменные и запишите в них:
·	features_zeros — признаки объектов с ответом «0»;
·	features_ones — признаки объектов с ответом «1»;
·	target_zeros — целевой признак объектов, у которых ответы только «0»;
·	target_ones — целевой признак объектов, у которых ответы только «1».
Напечатайте на экране размеры таблиц, которые хранятся в четырёх переменных (уже в прекоде).


import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

features_zeros = features_train[target_train == 0] 
features_ones = features_train[target_train == 1]
target_zeros = target_train[target_train == 0]
target_ones  = target_train[target_train == 1]

print(features_zeros.shape)
print(features_ones.shape)
print(target_zeros.shape)
print(target_ones.shape)

2.
Продублируйте объекты положительного класса и объедините их с объектами отрицательного класса. Чтобы соединить таблицы, воспользуйтесь функцией pd.concat() (от англ. concatenate, «сцепить»). Поработайте с документацией.
Мы объединили таблицы с признаками и сохранили результат в переменной features_upsampled (признаки, преобразованные техникой upsampling). Сделайте то же самое для целевого признака и объявите переменную target_upsampled (целевой признак, преобразованный техникой upsampling).
Количество повторений уже сохранено в переменной repeat (англ. «повторять»).
Напечатайте на экране размеры новых переменных (в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

features_zeros = features_train[target_train == 0]
features_ones = features_train[target_train == 1]
target_zeros = target_train[target_train == 0]
target_ones = target_train[target_train == 1]

repeat = 10
features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

print(features_upsampled.shape)
print(target_upsampled.shape)

3. Перемешайте данные. Импортируйте функцию shuffle() (англ. «перетасовать») из модуля sklearn.utils (от англ. «утилиты»).
Создайте функцию upsample() с тремя параметрами:
·	features — признаки,
·	target — целевой признак,
·	repeat — количество повторений.
Функция вернёт признаки и целевой признак после операции upsampling.
Вызовите функцию для обучающих данных. Если всё будет верно, размеры преобразованных выборок появятся на экране.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    return shuffle(features_upsampled,target_upsampled, random_state=12345)
    
# < добавьте перемешивание >

features_upsampled, target_upsampled = upsample(features_train, target_train, 10)

print(features_upsampled.shape)
print(target_upsampled.shape)

4.
Обучите на новых данных модель LogisticRegression. Найдите для неё значение F1-меры, и print() выведет её на экран.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 10)

model = LogisticRegression(random_state=12345,solver='liblinear')

model.fit(features_upsampled, target_upsampled)

predicted_valid = model.predict(features_valid)

score = model.score(features_valid, target_valid)

print("F1:", f1_score(target_valid, predicted_valid))


# Уменьшение выборки
1. Чтобы выполнить downsampling, напишите функцию downsample() с тремя аргументами:
·	features — признаки;
·	target — целевой признак;
·	fraction — доля отрицательных объектов, которые нужно сохранить.
Функция вернёт признаки и целевой признак после операции downsampling. Вызовите функцию для обучающих данных с аргументом fraction, равным 0.1. Код выведет на экран размеры выборок.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
  
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.1)

print(features_downsampled.shape)
print(target_downsampled.shape)

2. Обучите на новых данных модель LogisticRegression. Найдите для неё значение F1-меры, и код выведет его на экран.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.1)

model = LogisticRegression(random_state=12345, solver='liblinear')

model.fit(features_downsampled, target_downsampled)

predicted_valid = model.predict(features_valid)

score = model.score(features_valid, target_valid)
print("F1:", f1_score(target_valid, predicted_valid))

#Изменение порога
1.
Найдите значения вероятностей классов для валидационной выборки. Значения вероятностей класса «1» сохраните в переменной probabilities_one_valid. Напечатайте на экране первые пять элементов этой переменной (уже в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

# < напишите код здесь >
probabilities_valid = model.predict_proba(features_valid)  # < напишите код здесь >
probabilities_one_valid = probabilities_valid[:, 1]

print(probabilities_one_valid[:5])

2.
Переберите значения порогов от 0 до 0.3 с шагом 0.02. Найдите для каждого значения точность и полноту. Напечатайте результаты на экране (в прекоде).
Чтобы создать цикл с нужным диапазоном, мы применили функцию arange() (от англ. «упорядочивать») библиотеки numpy. Как и range(), функция перебирает указанные элементы диапазона, но работает не только с целыми, но и дробными числами. 
Средства библиотеки numpy вы ещё изучите.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.3, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)

    print("Порог = {:.2f} | Точность = {:.3f}, Полнота = {:.3f}".format(
        threshold, precision, recall))

    
    
    
    
# ROC-кривая

1.
Постройте ROC-кривую для логистической регрессии и изобразите её на графике. Вам поможет инструкция в коде. 
Для удобства мы уже добавили в код график ROC-кривой случайной модели.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve  # < напишите код здесь >


data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)  # < напишите код здесь >

plt.figure()

plt.plot(fpr, tpr)  # < постройте график >

# ROC-кривая случайной модели (выглядит как прямая)
plt.plot([0, 1], [0, 1], linestyle='--')


plt.xlim([0.0, 1.0])  # < примените функции plt.xlim() и plt.ylim(), чтобы
plt.ylim([0.0, 1.0])  #   установить границы осей от 0 до 1 >

plt.xlabel('False Positive Rate')  # < примените функции plt.xlabel() и plt.ylabel(), чтобы
plt.ylabel('True Positive Rate')  #   подписать оси "False Positive Rate" и "True Positive Rate" >

plt.title('ROC-кривая')  # < добавьте к графику заголовок "ROC-кривая" функцией plt.title() >

plt.show()


2.
Посчитайте для логистической регрессии AUC-ROC. В документации sklearn найдите, как функция расчёта этой площади называется и как она работает. Импортируйте её. Напечатайте  значение AUC-ROC на экране.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score  # < напишите код здесь >

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)  # < напишите код здесь >


print(auc_roc)


# Подготовка данных
1.
Загрузите данные из файла flights.csv. Напечатайте на экране:
размер таблицы;
первые пять строк таблицы.
Изучите данные.

import pandas as pd

data = pd.read_csv('/datasets/flights.csv')

# < напишите код здесь >
print(data.shape)
print(data.head(5))


2.
Преобразуйте категориальный признак техникой OHE. Приведите численные признаки к одному масштабу. Напечатайте на экране размеры таблиц (уже в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/datasets/flights.csv')

data_ohe = pd.get_dummies(data, drop_first=True)  # < преобразуйте данные так, чтобы избежать дамми-ловушки >
target = data_ohe['Arrival Delay']
features = data_ohe.drop(['Arrival Delay'] , axis=1)
# делим данные на обучающий и проверочный наборы
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)
# Нормализация числовых атрибутов
numeric = ['Day', 'Day Of Week', 'Origin Airport Delay Rate',
       'Destination Airport Delay Rate', 'Scheduled Time', 'Distance',
       'Scheduled Departure Hour', 'Scheduled Departure Minute']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])  # < преобразуйте валидационную выборку >

# Вывести размеры таблиц
print(features_train.shape)
print(features_valid.shape)


# MSE в задаче регрессии

1.
Загрузите данные из файла /datasets/flights_preprocessed.csv. Обучите линейную регрессию. Посчитайте значение MSE на валидационной выборке и сохраните его в переменной mse. 
Напечатайте на экране значения MSE и RMSE (уже в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)  # сохраните предсказания модели на валидационых данных
mse = mean_squared_error(target_valid, predicted_valid)

print("MSE =", mse)
print("RMSE =", mse ** 0.5)



2.
Вычислите MSE и RMSE для константной модели: каждому объекту она прогнозирует среднее значение целевого признака. Сохраните её предсказания в переменной predicted_valid. 
Напечатайте на экране значения MSE и RMSE (уже в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
mse = mean_squared_error(target_valid, predicted_valid)

print("Linear Regression")
print("MSE =", mse)
print("RMSE =", mse ** 0.5)

predicted_valid = pd.Series(target_train.mean(), index=target_valid.index)
mse = mean_squared_error(target_valid, predicted_valid) # < напишите код здесь >) 

print("Mean")
print("MSE =", mse)
print("RMSE =", mse ** 0.5)


# Коэффициент детерминации
Чтобы не пришлось всё время сравнивать модель со средним, введём новую метрику. Она выражена не в абсолютных значениях, а в относительных.
Коэффициент детерминации, или метрика R2 (англ. coefficient of determination; R-squared), вычисляет долю средней квадратичной ошибки модели от MSE среднего, а затем вычитает эту величину из единицы. Увеличение метрики означает прирост качества модели. 
Формула расчёта R2 выглядит так:


R2 = 1 - (MSE модели / MSE среднего)
Значение метрики R2 равно единице только в одном случае, если MSE нулевое. Такая модель предсказывает все ответы идеально.
R2 равно нулю: модель работает так же, как и среднее.
Если метрика R2 отрицательна, качество модели очень низкое.
Значения R2 больше единицы быть не может.

# задача
Вычислите для линейной регрессии значение R2. Найдите в документации sklearn.metrics функцию для подсчёта этой метрики. Импортируйте её. 
Напечатайте результат на экране (уже в прекоде).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # < напишите код здесь >

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print("R2 =", r2_score(target_valid, predicted_valid))  #  < напишите код здесь >)


# Среднее абсолютное отклонение
# MAE (mean absolute error)

1.
Напишите функцию mae() на основе формулы. На вход она принимает правильные ответы и предсказания. Возвращает значение среднего абсолютного отклонения. 
Обратите внимание: в Python модуль от числа вычисляется функцией abs() (от англ. absolute, «абсолютный»).
Проверьте работу функции на примере в прекоде. Напечатайте результат на экране.

import pandas as pd

def mae(target, predictions):
    error = 0
    for i in range(target.shape[0]):
        error += abs(target[i] - predictions[i])  # < напишите код здесь >)
    return error / target.shape[0]
            

target = pd.Series([-0.5, 2.1, 1.5, 0.3])
predictions = pd.Series([-0.6, 1.7, 1.6, 0.2])

print(mae(target, predictions))


Разбейте данные на обучающую и валидационную выборки.
Инициализируйте модель с гиперпараметрами лучшей модели из прошлого урока.
Обучите её на тренировочной выборке и посчитайте значение MAE на валидационной выборке.
Если на тестовой выборке значение MAE больше 26.2, вернитесь в предыдущий урок с Jupyter Notebook и продолжите экспериментировать. Или подсмотрите гиперпараметры в подсказке.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) # разбейте на обучающую и валидационную выборку

model = RandomForestRegressor(n_estimators=80, max_depth=11, random_state=12345) # выберите наилучшую модель
model.fit(features_train, target_train) # обучите модель на обучающей выборке
predictions_train = model.predict(features_train) # получите предсказания на обучающей выборке
predictions_valid = model.predict(features_valid) #получите предсказания на валидационной выборке
print("Наилучшая модель")
print("MAE на обучающей выборке: ", mean_absolute_error(target_train, predictions_train)) # найдите значение метрики MAE на обучающей выборке
print("MAE на валидационной выборке: ", mean_absolute_error(target_valid, predictions_valid)) # найдите значение метрики MAE на валидационной выборке
