# Оценка конечной геометрии

## Кратко

Оценщик сравнивает готовые M3D, не оценивает рецепт или процесс построения.
Объёмы и пересечение измеряются в живом КОМПАС, расстояния — по выгруженным поверхностям.
Рабочий пример: [KF0011, попытка №7](../runs/e2e/kf0011-codex-001/evaluation/comparison-001/report.md).
В этой паре выявлено несоответствие диаметра отверстия между PNG и эталонным M3D.

Оценщик работает вне папки `builder`. Эталоны и файлы оценки не передаются агенту построения.
Перед восстановлением эталона результат построителя фиксируется вместе с SHA256.
Файлы моделей открываются только оценочными копиями; открытые документы человека не закрываются.

Ссылки на `runs/`, исходные данные и результаты относятся к локальной рабочей копии;
эти файлы не входят в Git. См. [правила хранения](../docs/branches.md).

## Окружение

Отдельное окружение сохраняет тяжёлые зависимости вне основных зависимостей проекта:

```sh
UV_CACHE_DIR=/tmp/kompas-eval-cache uv venv --python 3.12 .venv-eval
UV_CACHE_DIR=/tmp/kompas-eval-cache uv pip install --python .venv-eval/bin/python -r evaluation/requirements.lock.txt
```

Windows: уже запущенный КОМПАС-3D v25 Study, Python и `ksapi.py` из его `Bin`,
`vendor/kompas_lib`. Настройки SSH и Windows-пути —
`runs/e2e/kf0011-codex-001/control/settings.json`.
`remote.py` использует `ssh windows_server` и передаёт PowerShell команду через UTF-16LE
`-EncodedCommand`. Нативные пробы запускаются последовательно. Нужен сетевой доступ доверенного
оценщика; ограничения shell построителя для этого не ослабляются.

## Последовательность

Для нового сравнения создаётся новая папка, содержащая `inputs/prediction.m3d`,
`inputs/reference.m3d` и манифест с хешами. Существующие чужие результаты не перезаписываются.
Команды ниже показывают воспроизведение уже сохранённого примера; переменные — только пути задачи.

```sh
eval_dir=runs/e2e/kf0011-codex-001/evaluation/comparison-001
eval_settings=runs/e2e/kf0011-codex-001/control/settings.json
eval_win_repo='C:\Mac\Home\Documents\VS_code_projects\ITMO\ASCON\kompas_auto_draft'
eval_win_dir="$eval_win_repo\runs\e2e\kf0011-codex-001\evaluation\comparison-001"
eval_bin='C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin'

python3 evaluation/remote.py "$eval_settings" "$eval_win_repo\evaluation\native_export.py" "$eval_win_dir" "$eval_bin"
.venv-eval/bin/python -m evaluation.geometry "$eval_dir" --native native-v2
python3 evaluation/remote.py "$eval_settings" "$eval_win_repo\evaluation\native_boolean.py" "$eval_win_dir" "$eval_bin"
MPLCONFIGDIR=/tmp/kompas-eval-mpl XDG_CACHE_HOME=/tmp/kompas-eval-cache .venv-eval/bin/python -m evaluation.visualize "$eval_dir"
```

`native_export.py` записывает МЦХ, ошибки дерева, текущие исполнения, STL со стрелкой 0,01 и 0,003 мм.
`geometry.py` сохраняет матрицу одного общего rigid-преобразования, расстояния, нормали,
топологию сетки, выборки и проверки на 20 000 / 80 000 точек. Масштаб и отражение запрещены.
Предполагаются одиночные детали; текущее исполнение сохраняется, переключения не выполняются.
`native_boolean.py` копирует тела в отдельную деталь и проверяет объёмы копий, преобразование и результат.
Сейчас нативный путь рассчитан на пару однокомпонентных пересекающихся тел. Сбой операции записывается
как ошибка оценщика; он не превращается в IoU=0. Пустые пересечения и сборки требуют расширения пути.

Дополнительные проверки KF0011:

```sh
python3 evaluation/remote.py "$eval_settings" "$eval_win_repo\evaluation\native_checks.py" "$eval_win_dir" "$eval_bin"
python3 evaluation/remote.py "$eval_settings" "$eval_win_repo\evaluation\native_boolean.py" "$eval_win_dir\sanity-overlap-cubes" "$eval_bin"
.venv-eval/bin/python -m evaluation.local_diagnostics "$eval_dir"
.venv-eval/bin/python -m evaluation.validate_exports "$eval_dir"
.venv-eval/bin/python -m evaluation.report "$eval_dir"
.venv-eval/bin/python -m unittest discover -s tests -p 'test_geometry_metrics.py' -v
```

`native_checks.py` читает ограниченный набор поверхностей эталона и создаёт контрольные кубы.
`local_diagnostics.py` предназначен именно для KF0011: подбирает окружность по вершинам сетки
отверстия и сопоставляет её с нативным цилиндром и вручную прочитанным размером на PNG.
`report.py` также формирует отчёт KF0011; для других деталей нужен отдельный слой интерпретации.
`validate_exports.py` сравнивает разрешения сеток при одном и том же преобразовании.

## Определения

- IoU = `Vintersection / (Vp + Vr - Vintersection)`; Dice = `2Vintersection / (Vp + Vr)`.
- Недостающий материал = `Vr - Vintersection`; лишний = `Vp - Vintersection`.
  Основная относительная нормировка обоих — на объём эталона `Vr`.
- Двустороннее среднее = полусумма двух средних расстояний до треугольников.
- Квадратичный Chamfer = полусумма средних квадратов этих расстояний, единицы мм²;
  RMS — его квадратный корень. Это не сумма без деления на два.
- Симметричный p95 = максимум p95 двух направлений.
- F1(τ): precision — доля площади результата в пределах τ от эталона;
  recall — доля площади эталона в пределах τ от результата; F1 — их гармоническое среднее.
- Максимум по выборке не является точным непрерывным Hausdorff. Нормали сравниваются
  в ближайших точках поверхностей; записаны как ориентированное, так и абсолютное скалярное произведение.
- Род границы вычисляется из эйлеровой характеристики замкнутой ориентированной сетки.
  Полной проверки самопересечений этот признак не заменяет.

Совмещение: главные оси инерции, многостартовый ICP, проверка симметричного расстояния,
уточнение L1. При неразличимости лучше сохранённые оси: граница выбора 0,0001 мм,
существенно меньше стрелки используемой сетки. Это фиксированная эвристика, не доказательство
глобального оптимума. Все сравнения используют одну выбранную матрицу.

Интерфейсы поверхностных расчётов: [trimesh.proximity](https://trimesh.org/trimesh.proximity.html),
[trimesh.registration](https://trimesh.org/trimesh.registration.html),
[trimesh.sample](https://trimesh.org/trimesh.sample.html).
Опорные источники KsAPI: `IAdditionConvertParameters`, `IMassInertiaParam`, `ICopyGeometry`,
`IBodyReposition`, `IBoolean`; использованы документация MCP `ksapi-expert` и обёртки установленной v25.

Инженерные пороги успешности пока не согласованы. Проверка отдельных требований чертежа,
полнота распознавания элементов, шероховатость и материал не подменяются геометрическим F1.
