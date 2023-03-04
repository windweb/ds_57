# Сравнение средних значений
'''
Интернет-магазин мармеладных червячков «Ползучая тянучка» разместил анимированного червяка-маскота на сайте. Нововведение «ползает» по экрану пользователя, пока тот выбирает товар. 
Посмотрите на две выборки средних чеков в интернет-магазине до и после запуска червячка. Напечатайте на экране значения средних до и после внедрения анимированного маскота. 
Проверьте гипотезу, что средний чек увеличился. Выберите уровень значимости, равный 5%. Выведите на экран значение p-value и результат проверки гипотезы.
'''
import pandas as pd
from scipy import stats as st

sample_before = pd.Series([
    436, 397, 433, 412, 367, 353, 440, 375, 414, 
    410, 434, 356, 377, 403, 434, 377, 437, 383,
    388, 412, 350, 392, 354, 362, 392, 441, 371, 
    350, 364, 449, 413, 401, 382, 445, 366, 435,
    442, 413, 386, 390, 350, 364, 418, 369, 369, 
    368, 429, 388, 397, 393, 373, 438, 385, 365,
    447, 408, 379, 411, 358, 368, 442, 366, 431,
    400, 449, 422, 423, 427, 361, 354])

sample_after = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print("Cреднее до:", sample_before.mean()) # < напишите код здесь >)
print("Cреднее после:", sample_after.mean()) # < напишите код здесь >)


# критический уровень статистической значимости
# если p-value окажется меньше него - отвергнем гипотезу
alpha = 0.05  # < напишите код здесь >)

results = st.ttest_ind(sample_before, sample_after) # передайте выборки до и после как аргументы функции)
    
# тест односторонний: p-value будет в два раза меньше
pvalue = results.pvalue / 2   # one-sided test # < напишите код здесь >)

print('p-значение: ', pvalue)

if pvalue < alpha:
    print("Отвергаем нулевую гипотезу: скорее всего средний чек увеличился")
else:
    print("Не получилось отвергнуть нулевую гипотезу: скорее всего средний чек не увеличился")
    
# ---
# Расчёт доверительного интервала

import pandas as pd
from scipy import stats as st

sample = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print("Cреднее:", sample.mean())

confidence_interval =  st.t.interval(  # < напишите код здесь >
    alpha=0.95,                   # confidence level (уровень доверия, равный единице минус уровень значимости)
    df=len(sample)-1,             # degrees of freedom — число степеней свободы, равное n-1.
    loc=sample.mean(),            # loc (от англ. location) — среднее распределение, равное оценке среднего
    scale=sample.sem()            # scale (англ. «масштаб») — стандартное отклонение распределения, равное оценке стандартной ошибки. можно записать так  scale=st.sem(sample)
)


print("95%-ый доверительный интервал:", confidence_interval)


# Бутстреп для доверительного интервала
1.
Процедурой бутстреп создайте 10 подвыборок и для каждой найдите 0.99-квантиль. Напечатайте их на экране через перенос строки.
Изучите функцию quantile() (англ. «квантиль значений») у объектов pandas.Series.


import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

for i in range(10):
    subsample = data.sample(frac=1, replace=True, random_state=state)
    print(subsample.quantile(0.99))  # < напишите код здесь >))



Функция quantile() в объектах pandas.Series используется для вычисления квантилей серии. Она принимает в качестве аргумента число с плавающей точкой q, где 0 <= q <= 1, и возвращает соответствующий квантиль серии. Например, series.quantile(0.5) возвращает медиану серии.

По умолчанию функция quantile() вычисляет квантиль выборки, используя метод, указанный в параметре q. Возможными методами являются:

    "linear": Интерполяция двух ближайших значений (по умолчанию).
    "lower": Выбор последнего наблюдения ниже квантиля.
    "higher": Выбрать первое наблюдение выше квантиля.
    "nearest": Выберите ближайшее к квантилю наблюдение.
    "midpoint": Среднее значение последнего наблюдения ниже и первого наблюдения выше квантиля.

Например, series.quantile(0.25, method="lower") возвращает нижний квартиль серии, используя метод "lower".


2.
Процедурой бутстреп найдите 90%-й доверительный интервал для 0.99-квантиля. Сохраните начало интервала в переменной lower, а конец — в upper. Код выведет их на экран.
Функцию quantile() вызовите дважды: сначала для получения 0.99-квантиля от каждой подвыборки, а потом для построения доверительного интервала.
    
import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

# сохраните значения 99%-квантилей в переменной values
values = []
for i in range(1000):
    subsample = data.sample(frac=1, replace=True, random_state=state)
    # < напишите код здесь >
    values.append(subsample.quantile(0.99))

# < напишите код здесь >
# вычислим 90% доверительный интервал для 0,99-квантиля
values = pd.Series(values)  # преобразуем тип для удобства
lower = values.quantile(0.05)  
upper = values.quantile(0.95)  

print(lower)
print(upper)


# Бутстреп для анализа A/B-теста
# Нулевая гипотеза предполагает равенство средних чеков в обеих группах. Альтернативная — в экспериментальной группе средний чек выше. Найдём p-value.
# Проанализируйте две выборки и проверьте гипотезу о том, что средний чек увеличился. Сохраните разность средних чеков в переменной AB_difference и напечатайте её на экране. Выберите уровень значимости, равный 5%. Сохраните в переменной pvalue значение p-value и напечатайте его на экране. Выведите на экран результат проверки гипотезы.

import pandas as pd
import numpy as np

# данные контрольной группы A
samples_A = pd.Series([
     98.24,  97.77,  95.56,  99.49, 101.4 , 105.35,  95.83,  93.02,
    101.37,  95.66,  98.34, 100.75, 104.93,  97.  ,  95.46, 100.03,
    102.34,  98.23,  97.05,  97.76,  98.63,  98.82,  99.51,  99.31,
     98.58,  96.84,  93.71, 101.38, 100.6 , 103.68, 104.78, 101.51,
    100.89, 102.27,  99.87,  94.83,  95.95, 105.2 ,  97.  ,  95.54,
     98.38,  99.81, 103.34, 101.14, 102.19,  94.77,  94.74,  99.56,
    102.  , 100.95, 102.19, 103.75, 103.65,  95.07, 103.53, 100.42,
     98.09,  94.86, 101.47, 103.07, 100.15, 100.32, 100.89, 101.23,
     95.95, 103.69, 100.09,  96.28,  96.11,  97.63,  99.45, 100.81,
    102.18,  94.92,  98.89, 101.48, 101.29,  94.43, 101.55,  95.85,
    100.16,  97.49, 105.17, 104.83, 101.9 , 100.56, 104.91,  94.17,
    103.48, 100.55, 102.66, 100.62,  96.93, 102.67, 101.27,  98.56,
    102.41, 100.69,  99.67, 100.99])

# данные экспериментальной группы B
samples_B = pd.Series([
    101.67, 102.27,  97.01, 103.46, 100.76, 101.19,  99.11,  97.59,
    101.01, 101.45,  94.8 , 101.55,  96.38,  99.03, 102.83,  97.32,
     98.25,  97.17, 101.1 , 102.57, 104.59, 105.63,  98.93, 103.87,
     98.48, 101.14, 102.24,  98.55, 105.61, 100.06,  99.  , 102.53,
    101.56, 102.68, 103.26,  96.62,  99.48, 107.6 ,  99.87, 103.58,
    105.05, 105.69,  94.52,  99.51,  99.81,  99.44,  97.35, 102.97,
     99.77,  99.59, 102.12, 104.29,  98.31,  98.83,  96.83,  99.2 ,
     97.88, 102.34, 102.04,  99.88,  99.69, 103.43, 100.71,  92.71,
     99.99,  99.39,  99.19,  99.29, 100.34, 101.08, 100.29,  93.83,
    103.63,  98.88, 105.36, 101.82, 100.86, 100.75,  99.4 ,  95.37,
    107.96,  97.69, 102.17,  99.41,  98.97,  97.96,  98.31,  97.09,
    103.92, 100.98, 102.76,  98.24,  97.  ,  98.99, 103.54,  99.72,
    101.62, 100.62, 102.79, 104.19])

# фактическая разность средних значений в группах
AB_difference = samples_B.mean() - samples_A.mean()  # < напишите код здесь >
print("Разность средних чеков:", AB_difference)

alpha = 0.05
    
state = np.random.RandomState(12345)

bootstrap_samples = 1000
count = 0
for i in range(bootstrap_samples):
    # объедините выборки
    united_samples = pd.concat([samples_A, samples_B]) 

    # создайте подвыборку
    subsample = united_samples.sample(frac=1, replace=True, random_state=state)  # < напишите код здесь >
    
    # разбейте подвыборку пополам
    subsample_A = subsample[:len(samples_A)]
    subsample_B = subsample[len(samples_A):]

    # найдите разницу средних
    bootstrap_difference = subsample_B.mean() - subsample_A.mean()  # < напишите код здесь >
    
    # если разница не меньше фактической, увеличиваем счётчик
    if bootstrap_difference >= AB_difference:
        count += 1

# p-value равно доле превышений значений
pvalue = 1. * count / bootstrap_samples
print('p-value =', pvalue)

if pvalue < alpha:
    print("Отвергаем нулевую гипотезу: скорее всего, средний чек увеличился")
else:
    print("Не получилось отвергнуть нулевую гипотезу: скорее всего, средний чек не увеличился")

    
    
# Бутстреп для моделей
'''
How to use Bootstrap to estimate confidence intervals for machine learning models?
Zaspuk in 5 Minutes, a school for express English classes, is developing a model to estimate the probability that a student will come to class or not. Many applications are received every day. Priority is given on a first-come, first-served basis. About half of the students don't show up and don't pay for the class. The school management has decided to keep classes only for those students who are most likely to come. Because of the possible reputational risks, the company will implement this innovation if there is a guarantee of a noticeable increase in revenue. To make the right decision, estimate the probability distribution for profits.
Here are the important terms of the task:
The model for predicting the probability of attending a class has already been trained. The predictions are in eng_probabilities.csv, and the correct answers are in eng_target.csv.
One lesson costs 1000 rubles. You can schedule up to 10 lessons per day. Current revenue for the day - 5000 rubles (half of the students cancel the lesson).
Per day there is an average of 25 applications.
Enough revenue to implement - 7500 rubles. Its probability should be at least 99%.

The task is . 1.
Write a function revenue(), which calculates and returns the revenue. It receives as input:
a list of target responses - whether the student came to class;
a list of probabilities probabilities - the model estimates whether the student is coming or not;
how many students attend in a day count.
The function should select the students with the highest probability of attendance and, based on the answers, calculate the possible revenue.
The pre-code is an example of running the function when the lists of probabilities and answers are small and there are only three students.
'''
'''
1.
Напишите функцию revenue() (англ. «выручка»), которая подсчитывает и возвращает выручку. Она получает на вход:
список ответов target — пришёл ли ученик на урок;
список вероятностей probabilities — модель оценивает, придёт ученик или нет;
сколько учеников посещает занятия за день count.
Функция должна выбрать учеников с наибольшей вероятностью посещения и на основе ответов подсчитать возможную выручку.
В прекоде пример запуска функции, когда списки вероятностей и ответов невелики, а  учеников только трое.
'''
import pandas as pd

def revenue(target, probabilities, count):
    # < напишите код здесь >
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    revenue = selected.sum()
    return 1000 * revenue # < напишите код здесь >

target = pd.Series([1,   1,   0,   0,  1,    0])
probab = pd.Series([0.2, 0.9, 0.8, 0.3, 0.5, 0.1])

res = revenue(target, probab, 3)

print(res)

'''
2.
Чтобы найти 1%-квантиль выручки, выполните процедуру Bootstrap с 1000 повторений. 
Сохраните список оценок из бутстрепа в переменной values, а 0.01-квантиль — в переменной lower. Код выведет на экран среднюю выручку и 0.01-квантиль.

Переменная subsample создается путем вызова метода sample() с аргументом n, установленным в 25, чтобы получить подвыборку из 25 студентов. Аргумент replace=True устанавливается для включения выборки с заменой.
Затем переменная probs_subsample получается путем выбора вероятностей, соответствующих индексам подвыборки. Наконец, доход рассчитывается с помощью функции revenue() с учетом вероятностей подвыборки и подвыборки.
Значения добавляются в список значений на каждой итерации, а квантиль 1% получается с помощью метода quantile().
Переведено с помощью www.DeepL.com/Translator (бесплатная версия)
'''

import pandas as pd
import numpy as np

# открываем файлы
# возьмём индекс '0', чтобы перевести данные в pd.Series
target = pd.read_csv('/datasets/eng_target.csv')['0']
probabilities = pd.read_csv('/datasets/eng_probabilities.csv')['0']

def revenue(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    return 1000 * selected.sum()

state = np.random.RandomState(12345)
    
values = []
for i in range(1000):
    # < напишите код здесь>
    subsample = target.sample(n=25, random_state=state, replace=True)
    probs_subsample = probabilities[subsample.index]
    rev = revenue(subsample, probs_subsample, 10)
    values.append(rev)  # < напишите код здесь>)

values = pd.Series(values)
lower = values.quantile(0.01)  # < напишите код здесь>

mean = values.mean()
print("Средняя выручка:", mean)
print("1%-квантиль:", lower)

