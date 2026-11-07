"""Сводка метрик и отчёт из сохранённых измерений; КОМПАС не вызывается."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.geometry import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    path = parser.parse_args().directory
    native = {
        label: json.loads((path / "native-v2" / f"{label}.json").read_text())
        for label in ("prediction", "reference")
    }
    p, r = native["prediction"], native["reference"]
    surface = json.loads((path / "surface-metrics.json").read_text())
    boolean = json.loads((path / "native-boolean.json").read_text())
    local = json.loads((path / "bore-measurements.json").read_text())
    tess = json.loads((path / "tessellation-check.json").read_text())
    sanity = json.loads((path / "sanity-overlap-cubes/native-boolean.json").read_text())
    if boolean["status"] != "ok" or sanity["status"] != "ok":
        raise ValueError("Native boolean failed; do not publish an IoU")
    if not np.isclose(sanity["iou"], 1 / 3, atol=1e-10):
        raise ValueError("Control cubes did not produce the expected IoU")
    matrix = np.array(surface["alignment"]["matrix_prediction_to_reference"])
    if not np.allclose(matrix, boolean["transform"], atol=1e-10):
        raise ValueError("Surface and native metrics used different transforms")
    for label in native:
        sha = hashlib.sha256((path / "inputs" / f"{label}.m3d").read_bytes()).hexdigest()
        if sha != native[label]["file_sha256"]:
            raise ValueError("Snapshot hash mismatch")
    sm = surface["surface_80000"]
    ref_box = np.array(r["bbox_native"][1:]).reshape(2, 3)
    pred_box = np.array(p["bbox_native"][1:]).reshape(2, 3)
    diagonal = float(np.linalg.norm(ref_box[1] - ref_box[0]))
    center = matrix[:3, :3] @ np.array(p["center_mm"]) + matrix[:3, 3]
    center_distance = float(np.linalg.norm(center - np.array(r["center_mm"])))

    def principal_gyration(data: dict) -> list[float]:
        x, y, z, xy, xz, yz = data["inertia_native"]
        tensor = np.array([[x, -xy, -xz], [-xy, y, -yz], [-xz, -yz, z]]) / data["mass_native"]
        return np.linalg.eigvalsh(tensor).tolist()

    gp, gr = map(principal_gyration, (p, r))
    bulk = {
        "prediction_volume_mm3": p["volume_mm3"],
        "reference_volume_mm3": r["volume_mm3"],
        "volume_signed_relative_error": p["volume_mm3"] / r["volume_mm3"] - 1,
        "prediction_area_mm2": p["area_mm2"],
        "reference_area_mm2": r["area_mm2"],
        "area_signed_relative_error": p["area_mm2"] / r["area_mm2"] - 1,
        "prediction_extents_mm": (pred_box[1] - pred_box[0]).tolist(),
        "reference_extents_mm": (ref_box[1] - ref_box[0]).tolist(),
        "center_distance_after_alignment_mm": center_distance,
        "center_distance_over_reference_diagonal": center_distance / diagonal,
        "prediction_principal_gyration_mm2": gp,
        "reference_principal_gyration_mm2": gr,
        "principal_gyration_relative_error": (np.array(gp) / np.array(gr) - 1).tolist(),
    }
    boolean["extra_fraction_reference"] = boolean["extra_volume_mm3"] / r["volume_mm3"]
    quality = {}
    for label in native:
        q = surface[label]
        quality[label] = {
            "native_bodies": native[label]["bodies"],
            "native_solids": native[label]["solid"],
            "active_tree_errors": [i for i in native[label]["tree"] if i["error"] and not i["excluded"]],
            "mesh_watertight": q["watertight"],
            "mesh_winding_consistent": q["winding_consistent"],
            "boundary_shells": q["boundary_shells"],
            "boundary_genus_sum": q["boundary_genus_sum"],
            "betti_numbers_from_closed_boundary": [
                native[label]["bodies"],
                q["boundary_genus_sum"],
                q["boundary_shells"] - native[label]["bodies"],
            ],
            "mesh_volume_relative_error": q["mesh_volume_mm3"] / native[label]["volume_mm3"] - 1,
        }
    validation = {
        "sample_count_mean_delta_mm": sm["symmetric_mean_mm"] - surface["surface_20000"]["symmetric_mean_mm"],
        "sample_count_p95_delta_mm": sm["symmetric_p95_mm"] - surface["surface_20000"]["symmetric_p95_mm"],
        "tessellation_mean_delta_mm": tess["chord_0.003_mm"]["symmetric_mean_mm"]
        - tess["chord_0.01_mm"]["symmetric_mean_mm"],
        "tessellation_p95_delta_mm": tess["chord_0.003_mm"]["symmetric_p95_mm"]
        - tess["chord_0.01_mm"]["symmetric_p95_mm"],
        "control_cubes_iou": sanity["iou"],
        "control_cubes_expected_iou": 1 / 3,
    }
    result = {
        "schema_version": 1,
        "pair_id": "KF0011",
        "selected_attempt": 7,
        "status": "measured_reference_requires_review",
        "geometrically_equal": False,
        "engineering_success": None,
        "reason": "Reference bore differs from drawing; acceptance not agreed",
        "prediction_sha256": p["file_sha256"],
        "reference_sha256": r["file_sha256"],
        "byte_identical": p["file_sha256"] == r["file_sha256"],
        "tree_identical": p["tree"] == r["tree"],
        "isolation": "not_attested_for_manual_desktop_session",
        "rigid_transform": matrix.tolist(),
        "bulk": bulk,
        "volume_overlap": boolean,
        "surface": sm,
        "surface_mean_over_reference_diagonal": sm["symmetric_mean_mm"] / diagonal,
        "validity_and_topology": quality,
        "local_diagnostics": local,
        "numerical_validation": validation,
        "not_measured": [
            "exact_continuous_Hausdorff",
            "full_self_intersection_certificate",
            "engineering_feature_coverage",
            "drawing_tolerance_acceptance",
        ],
    }
    save_json(path / "metrics.json", result)
    f1 = next(row["f1"] for row in sm["thresholds"] if row["tolerance_mm"] == 0.1)
    rows = "\n".join(
        f"| {v['tolerance_mm']:g} | {v['precision'] * 100:.2f}% | "
        f"{v['recall'] * 100:.2f}% | {v['f1'] * 100:.2f}% |"
        for v in sm["thresholds"]
    )
    report = f"""# KF0011 — сравнение попытки №7 с M3D пары

## Кратко

Модель агента похожа на M3D пары, но геометрически ему не равна: **IoU {boolean["iou"] * 100:.2f}%**,
среднее расстояние поверхностей **{sm["symmetric_mean_mm"]:.3f} мм**,
**F1@0,1 мм {f1 * 100:.2f}%**. Расчёт объёмов и пересечения выполнен в КОМПАС-3D v25 Study,
сборка 2735. Обе модели содержат одно твёрдое тело без ошибок активных элементов дерева.

**В паре данных обнаружено противоречие:** на входном PNG отверстие Ø32 H8 (+0,039),
а в эталонном M3D — Ø30,6 мм. Геометрия результата агента даёт ≈Ø32 мм.
Основная ошибка относительно M3D поэтому не доказывает ошибку агента относительно чертежа.
Эталон оставлен неизменным. Итог «успех/неуспех» не присваивается до проверки пары и согласования допусков.

## Файлы и происхождение

- [Модель агента](../../artifacts/model.m3d), выбранная [попытка №7](../../artifacts/selection.json).
- [Входной PNG](../../builder/input/drawing.png).
- Неизменяемые оценочные копии: [prediction](inputs/prediction.m3d), [reference](inputs/reference.m3d).
- Эталон восстановлен из `kompas-drawings/data/kompas-files/pairs/KF0011/model.m3d`;
  его SHA256 совпал с замороженным манифестом `e2e-27-v1`. Проверены хеши всех эталонов набора.
- [Полные числа](metrics.json), [измерения КОМПАС](native-v2/reference.json),
  [контроль пересечения](native-boolean.json), [локальная проверка отверстия](bore-measurements.json).

SHA256 результата: `{p["file_sha256"]}`.\x20\x20
SHA256 эталона: `{r["file_sha256"]}`.

Хеши, деревья и комментарии документов различаются. Это не доказывает изоляцию построителя:
для ручного запуска в desktop-чате техническая изоляция не аттестована.
Метрики описывают этот тестовый результат и не объявляются чистым слепым бенчмарком.

## Объёмное сходство

| Показатель | Значение |
|---|---:|
| IoU = V∩ / V∪ | {boolean["iou"]:.8f} |
| Dice = 2V∩ / (Vp + Vr) | {boolean["dice"]:.8f} |
| Объём результата | {p["volume_mm3"]:.3f} мм³ |
| Объём эталона | {r["volume_mm3"]:.3f} мм³ |
| Относительное отличие объёма | {bulk["volume_signed_relative_error"] * 100:+.3f}% |
| Объём пересечения | {boolean["intersection_mm3"]:.3f} мм³ |
| Объём объединения, Vp + Vr − V∩ | {boolean["union_mm3"]:.3f} мм³ |
| Недостающий материал | {boolean["missing_volume_mm3"]:.3f} мм³ |
| Недостающий материал / Vr | {boolean["missing_fraction_reference"] * 100:.3f}% |
| Лишний материал | {boolean["extra_volume_mm3"]:.3f} мм³ |
| Лишний материал / Vr | {boolean["extra_fraction_reference"] * 100:.3f}% |
| Симметрическая разность | {boolean["symmetric_difference_mm3"]:.3f} мм³ |

«Недостающий» и «лишний» относятся к сравнению с M3D, а не к требованиям PNG.
Пересечение выполнено нативной операцией `IBoolean` на копиях тел в отдельном документе.
Сохранены [тела перед пересечением](boolean-inputs.m3d) и [результат пересечения](intersection.m3d).
Воксели и стороннее B-rep ядро не использовались.

## Поверхностное сходство

| Показатель | Значение |
|---|---:|
| Среднее симметричное расстояние | {sm["symmetric_mean_mm"]:.6f} мм |
| RMS | {sm["symmetric_rms_mm"]:.6f} мм |
| Chamfer: полусумма средних квадратов | {sm["chamfer_squared_mean_mm2"]:.6f} мм² |
| Симметричный p95: максимум двух направленных p95 | {sm["symmetric_p95_mm"]:.6f} мм |
| Максимум на выборке | {sm["sampled_hausdorff_lower_bound_mm"]:.6f} мм |
| Согласованность нормалей | {sm["normal_consistency_absolute"]:.6f} |

По 80 000 точек на каждой поверхности, равномерно по площади, включая внутренние поверхности.
Расстояние измеряется до треугольников другой поверхности, а не до другой случайной выборки точек.
Для нормалей берётся среднее абсолютного скалярного произведения в ближайших точках.
Максимум на выборке — нижняя оценка непрерывного Hausdorff для сеток, не гарантированный максимум B-rep.

| Допуск, мм | Precision: результат → эталон | Recall: эталон → результат | F1 |
|---|---:|---:|---:|
{rows}

Это кривая диагностических допусков, не принятые пороги успешности. F1 — гармоническое среднее
двух долей площади, попавших в допуск. F1=100% при 1 мм не означает инженерной правильности.

## Габариты, МЦХ и топология

- Площади: результат {p["area_mm2"]:.3f} мм², эталон {r["area_mm2"]:.3f} мм²;
  отличие {bulk["area_signed_relative_error"] * 100:+.3f}%.
- Размеры по XYZ: результат `{bulk["prediction_extents_mm"]}`, эталон `{bulk["reference_extents_mm"]}` мм.
- Расстояние центров после совмещения: {center_distance:.6f} мм.
- Главные центральные моменты, делённые на массу (мм²), и их относительные отличия записаны в
  `metrics.json → bulk`. Они получены из МЦХ КОМПАС; нормировка убирает различие плотностей материалов.
- Обе сетки замкнуты, с согласованной ориентацией; одна оболочка и род 1.
  Производные числа Бетти тела `(b0,b1,b2)=(1,1,0)`: одно тело, один сквозной канал, без замкнутой полости.
- Число B-rep граней различается: {p["faces"]} против {r["faces"]}. Это не штраф за историю построения.
  Полной сертификации отсутствия самопересечений и отдельной проверки всех элементов по допускам нет.

## Совмещение и устойчивость

Одно преобразование для всех метрик: **перенос результата на +16,5 мм по Z**, без масштаба,
отражения и деформации. Многостартовый поиск с ICP проверял главные оси и вращение вокруг оси шестерни;
окончательный критерий — среднее двустороннее расстояние до поверхности.
Микроповорот улучшал оценку лишь примерно на 0,000013 мм, поэтому выбран вариант с сохранением осей:
разница меньше зафиксированной границы выбора 0,0001 мм и точности триангуляции.
[Все кандидаты и правило выбора](alignment.json) сохранены. Глобальный оптимум математически не доказан.

Экспорт STL с параметром стрелки 0,003 мм и углом 0,05; единицы явно заданы в миллиметрах.
Несмотря на флаг binary, КОМПАС записал ASCII STL — файлы проверены парсером.
Отклонение объёма сетки от B-rep: результат {quality["prediction"]["mesh_volume_relative_error"] * 100:.5f}%,
эталон {quality["reference"]["mesh_volume_relative_error"] * 100:.5f}%.

При увеличении выборки с 20 000 до 80 000 среднее изменилось на
{validation["sample_count_mean_delta_mm"]:+.6f} мм,
p95 — на {validation["sample_count_p95_delta_mm"]:+.6f} мм.
При уточнении стрелки с 0,01 до 0,003 мм на одинаковых размерах выборки среднее изменилось на
{validation["tessellation_mean_delta_mm"]:+.6f} мм,
p95 — на {validation["tessellation_p95_delta_mm"]:+.6f} мм.
Это проверки устойчивости, не строгие границы погрешности; самый малый допуск 0,01 мм чувствителен к сетке.

Нативный оценщик проверен на двух кубах 10×10×10 мм со сдвигом 5 мм:
пересечение 500 мм³, объединение 1500 мм³, IoU = 1/3.
При микроповороте шестерни КОМПАС возвращал ошибку пересечения 5; неудачные диагностики
сохранены в `initial-calculation/`. Устойчивая поза с сохранением осей дала успешную операцию.

## Иллюстрации

![Поверхности, сечения и отклонения](comparison.png)

![Покрытие поверхности по допуску](coverage.png)

## Что следует из результата

Оценщик различает общий объём и локальную ошибку и обнаружил проблему в паре данных.
KF0011 следует пометить как требующий проверки соответствия PNG ↔ M3D перед использованием
в итоговой оценке качества генерации. Автоматически исправлять эталон под результат агента нельзя.
Полное инженерное соответствие зубчатого профиля и всех требований чертежа этим расчётом не установлено.

Воспроизведение и определения: [инструкция оценщика](../../../../../evaluation/README.md).
"""
    temporary = path / "report.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path / "report.md")
    print(
        json.dumps(
            {
                "iou": boolean["iou"],
                "mean_mm": sm["symmetric_mean_mm"],
                "f1_0.1": f1,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
