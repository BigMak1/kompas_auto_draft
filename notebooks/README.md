# Анализ набора «чертёж — модель»

## Кратко

[Ноутбук](EDA.ipynb) содержит обзор 10 пар и подробный разбор шестерни,
прорезной втулки и кольца с отверстиями. Все детали относятся к валидации.
В нём сохранены изображения, таблицы и результаты выполнения.
Выводы находятся в последнем разделе ноутбука.
В разделе моделей есть интерактивный просмотр STL: выбор детали, вращение,
масштабирование и координаты в миллиметрах. Статическая галерея сохранена.

## Окружение EDA и просмотра CAD

У проекта два отдельных окружения на Python 3.12:

| Проект | Окружение | Назначение |
|---|---|---|
| `kompas_auto_draft` | `.venv` в корне | Агент, построение baseline и оценка |
| `kompas-hw01-eda` | `notebooks/.venv` | Jupyter, анализ, Plotly и OCP CAD Viewer |

Зависимости EDA и просмотра закреплены в `notebooks/pyproject.toml` и `notebooks/uv.lock`.
Здесь установлены CadQuery 2.7, OCP 7.8 и `ocp-vscode` 4.2, совместимый с расширением
OCP CAD Viewer 4.2 в VS Code. Внутри ноутбука готовые STL отображаются через
[Plotly Mesh3d](https://plotly.com/python/3d-mesh/) и виджеты Jupyter.
Геометрия и результаты baseline от просмотра не меняются.

## Запуск

Из корня репозитория установите зависимости и зарегистрируйте ядро в созданном окружении:

```bash
uv sync --project notebooks --locked
uv run --project notebooks --locked python -m ipykernel install --sys-prefix --name kompas-eda --display-name "EDA и 3D (Python 3.12)"
uv run --project notebooks --locked jupyter lab notebooks/EDA.ipynb
```

В ноутбуке выберите **«EDA и 3D (Python 3.12)»**, перезапустите ядро и выполните все ячейки.
Ядро использует Python из `notebooks/.venv`. Зависимости закреплены в `notebooks/uv.lock`.
В VS Code можно выбрать это окружение непосредственно в меню выбора ядра ноутбука.
Для переключения деталей в интерактивном просмотре ядро должно работать;
GitHub показывает сохранённые таблицы и статические изображения.

Выполнить все ячейки без браузера и сохранить результаты можно стандартной командой Jupyter:

```bash
uv run --project notebooks --locked jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb
```

Для анализа достаточно файлов репозитория. КОМПАС и доступ к модели генерации не нужны.
Исходные изображения лежат в `data/input`, эталонные STL — в `data/target`.
CDW, исходные сетки и измерения собраны по деталям в `data/supporting`.
Различия исходных пар KF0011, KF0012 и KF0018 описаны в анализе и описи.
У KF0018 отдельно отмечено, что диаметры находятся в указанных допусках,
а осевой габарит и профиль отличаются от изображения на листе.

## OCP CAD Viewer в VS Code

Для экспериментов с объектами CadQuery выберите **Python: Select Interpreter →
Enter interpreter path** и Python из `notebooks/.venv` (`bin/python` на macOS/Linux,
`Scripts/python.exe` на Windows). После смены окружения перезапустите просмотрщик,
открытое ядро или Python-консоль, чтобы они использовали новое окружение.
В панели OCP CAD Viewer откройте просмотрщик; затем можно выполнить:

```python
import cadquery as cq
from ocp_vscode import show

show(cq.Workplane("XY").box(1, 2, 3))
```

Команды baseline по-прежнему запускаются из корня через `uv run poe …` и используют
корневое окружение. Для библиотек EDA используйте `uv add --project notebooks …`
и сохраняйте обновлённый lock-файл; для повторной установки — `uv sync --project notebooks --locked`.

Уже обнаруженный конфликт возник после установки `build123d`: пакет `cadquery-ocp-novtk`
версии 8 записал свой модуль `OCP` поверх `cadquery-ocp` 7.8, нужного baseline.
Поэтому в этих окружениях используется один поставщик `OCP`, зафиксированный через uv.
Кнопки Quickstart и обновления библиотек в расширении запускают собственные команды установки
и обходят наш lock-файл ([документация Library Manager](https://bernhard-42.github.io/ocp_viewer_docs/viewers/ocp_vscode/managers/)).
Для этого проекта устанавливайте библиотеки приведёнными командами uv.
