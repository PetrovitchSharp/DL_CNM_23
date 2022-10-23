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



## Сравнение моделей

| Model | Recall (macro) | Precision (macro) | F1 (macro) |
| --- | --- | --- | --- |
| Random Forest | 0.70 | 0.95 | 0.78 |
| CatBoost | 0.77 | 0.90 | 0.83 |
| LightGBM | 0.78 | 0.89 | 0.82 |
| MLP | 0.52 | 0.87 | 0.54 |

## Использование нейронных сетей (LSTM)

<p>При использовании нейронных сетей мы переформулировали задачу из задачи классификации пар названий компаний (как в датасете) в задачу поиска похожих компаний по базе. То есть пользователь вводит название компании и ожидает, что система выдаст ему список из нескольких похожих компаний, чтобы пользователь мог понять "есть ли уже данная компания в базе данных и выбрать ее.</p><br>

<p> Для этого нам пришлось выполнить следующие шаги:</p>

1. Кластеризация данных (там где это возможно) -нужно запустить ноутбук notebooks/2.0-Data-clusterization.ipynb
1. Подготовка датасета для обучения нейронных сетей - для этого нужно выполнить команду:

    poetry run python data/split_dataset.py

3. Обучение модели - ноутбук notebooks/3.0-LSTM-train.ipynb
4. Оценка качества модели - ноутбук notebooks/3.1-validation.ipynb

## Результат

<p>На данный момент я попробовал обучать сеть на основе LSTM (данные подаются посимвольно), было проведено достаточно много экспериментов с различными гиперпараметрами (embedding_size, hidden_size, количество слоев LSTM, различные значения dropout.</p><br>

