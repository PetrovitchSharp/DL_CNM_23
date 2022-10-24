DL_CNM_23
==============================

Сопоставление названий компаний

Структура проекта
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

## Метрики

В качестве основных метрик качества классификации были выбраны [recision и recall](https://en.wikipedia.org/wiki/Precision_and_recall), а также [F1-score](https://en.wikipedia.org/wiki/F-score) в macro интерпретации (т.е. как среднее арифметическое по данным метрикам для каждого из классов). 

Precision можно интерпретировать как долю объектов, названных классификатором положительными и при этом действительно являющимися положительными, а recall показывает, какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм. Recall демонстрирует способность алгоритма обнаруживать данный класс вообще, а precision — способность отличать этот класс от других классов. 

![Precision](https://habrastorage.org/getpro/habr/post_images/164/93b/c89/16493bc899f7275f3b5ff8d45a3ed2e2.svg)

![Recall](https://habrastorage.org/getpro/habr/post_images/258/4e7/8f3/2584e78f32225eade5cb8b1b4a665193.svg)


F1-score - это метрика, объединяющая в себе информацию о точности (precision) и полноте (recall) модели.

![F1](https://miro.medium.com/max/1400/1*9uo7HN1pdMlMwTbNSdyO3A.png)



## Использование классического ML

Нами было решено опробовать для решения задачи классификации названий компаний такие модели, как [random forest](https://github.com/PetrovitchSharp/DL_CNM_23/blob/develop/notebooks/random_forest.ipynb), [CatBoost, LightGBM](https://github.com/PetrovitchSharp/DL_CNM_23/blob/develop/notebooks/boosting_models.ipynb), а также сравнить их с простейшим [многослойным персептроном](https://github.com/PetrovitchSharp/DL_CNM_23/blob/develop/notebooks/neural_network.ipynb).

Для обучения моделей был произведен препроцессинг ([ноутбук раз](https://github.com/PetrovitchSharp/DL_CNM_23/blob/develop/notebooks/text_unification.ipynb) и [ноутбук два](https://github.com/PetrovitchSharp/DL_CNM_23/blob/develop/notebooks/feature_extraction.ipynb)) с целью вычленить числовые фичи из названий.

В результате удалось метрик, представленных в таблице ниже.

| Model | Recall (macro) | Precision (macro) | F1 (macro) |
| --- | --- | --- | --- |
| Random Forest | 0.70 | 0.95 | 0.78 |
| CatBoost | 0.77 | 0.90 | 0.83 |
| LightGBM | 0.78 | 0.89 | 0.82 |
| MLP | 0.52 | 0.87 | 0.54 |

В результате в качестве опорной модели из класса "классических" был выбран CatBoost.

## Использование нейронных сетей (LSTM)

<p>При использовании нейронных сетей мы переформулировали задачу из задачи классификации пар названий компаний (как в датасете) в задачу поиска похожих компаний по базе. То есть пользователь вводит название компании и ожидает, что система выдаст ему список из нескольких похожих компаний, чтобы пользователь мог понять "есть ли уже данная компания в базе данных и выбрать ее.</p><br>

<p> Для этого нам пришлось выполнить следующие шаги:</p>

1. Кластеризация данных (там где это возможно) -нужно запустить ноутбук notebooks/2.0-Data-clusterization.ipynb
1. Подготовка датасета для обучения нейронных сетей - для этого нужно выполнить команду:

    poetry run python data/split_dataset.py

3. Обучение модели - ноутбук notebooks/3.0-LSTM-train.ipynb
4. Оценка качества модели - ноутбук notebooks/3.1-validation.ipynb

### Результат

<p>На данный момент мы попробовали обучать сеть на основе LSTM (данные подаются посимвольно), было проведено достаточно много экспериментов с различными гиперпараметрами (embedding_size, hidden_size, количество слоев LSTM, различные значения dropout.</p><br>

![Wandb](https://raw.githubusercontent.com/PetrovitchSharp/DL_CNM_23/feature/lstm_model/img/wandb_lstm.png)

### Метрика

<p>В этой задаче для пользователя важно, чтобы среди предложенных моделью TOP K результатов был нужный. Мы взяли в качестве K - 5. Для каждого кластера из валидационного датасета был рассчитан precision (доля элементов этого кластера, при поиске по которым среди TOP 5 ответов есть элементы этого кластера).</p>

| Model | Precision (clusters) |
| --- | --- |
| LSTM (emb_size 80, hidden_size 80, layers 3) character level | 0.60 |

## Запуск пайплайна

### CatBoost

1. Скачайте [исходный датасет](https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view?usp=sharing) и распакуйте его в папку data/raw

2. Выполните предобработку данных - src/features/build_catboost_features.py

    Параметры для использования скрипта:

        -h, --help            show this help message and exit
        -data DATA            raw dataset filename
        -output OUTPUT        prepared dataset filename

3. Обучите модель CatBoost - scr/models/train_catboost.py 

    Параметры для использования скрипта:

        -h, --help            show this help message and exit
        -data DATA            preprocessed dataset filename
        -version VERSION      model version
        -lr LR                learning rate
        -iters ITERS          number of iterations
        -test_size TEST_SIZE  ratio of test part
        -refit REFIT          refit model on full dataset

4. Запустите инференс модели - scr/models/catboost_inference.py 

    Параметры для использования скрипта:

        -h, --help    show this help message and exit
        -input INPUT  json file with company names
        -model MODEL  model filename
