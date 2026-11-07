# Анализ трёх пар «чертёж — модель»

## Кратко

[Ноутбук](EDA.ipynb) разбирает шестерню, прорезную втулку и кольцо с отверстиями.
В нём сохранены изображения, таблицы и результаты выполнения.
Выводы находятся в последнем разделе ноутбука.

## Запуск

Из корня репозитория установите зависимости и зарегистрируйте ядро в созданном окружении:

```bash
uv sync --project notebooks --locked
uv run --project notebooks --locked python -m ipykernel install --sys-prefix --name kompas-eda --display-name "Анализ чертежей (Python)"
uv run --project notebooks --locked jupyter lab notebooks/EDA.ipynb
```

В ноутбуке выберите **«Анализ чертежей (Python)»**, перезапустите ядро и выполните все ячейки.
Ядро использует Python из `notebooks/.venv`. Зависимости закреплены в `notebooks/uv.lock`.

Выполнить все ячейки без браузера и сохранить результаты можно стандартной командой Jupyter:

```bash
uv run --project notebooks --locked jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb
```

Для анализа достаточно файлов репозитория. КОМПАС и доступ к модели генерации не нужны.
Исходные изображения лежат в `data/input`, модели и их выгрузки — в `data/reference`.
