# Оценка CAD-моделей

## Кратко

Основные показатели — объёмный IoU и F-score. Остальные — вспомогательные.
Протокол: `cad-eval-0.2.0`. Назначение: `benchmark`.
Полнота основных метрик: да.

| Деталь | Статус | IoU ↑ | F-score ↑ |
|---|---|---:|---:|
| KF0017 | evaluated | 1 | 1 |
| KF0027 | evaluated | 1 | 1 |

Основные средние по всем попыткам (нули за невалидную генерацию):

- f_score: 1; измерено 2/2.
- solid_iou: 1; измерено 2/2.

## Вспомогательные метрики

| Показатель | Среднее | Медиана | Учтено попыток |
|---|---:|---:|---:|
| chamfer_l2_reference_normalized | 7.2792e-05 | 7.2792e-05 | 2 |
| dangling_edge_length_mm | 0 | 0 | 2 |
| dimension_absolute_error_deg_mean | не считалось | не считалось | 0 |
| dimension_absolute_error_mm_mean | не считалось | не считалось | 0 |
| dimension_absolute_error_ratio_mean | не считалось | не считалось | 0 |
| dimension_compliance | не считалось | не считалось | 0 |
| dimension_deg_measured_count | 0 | 0 | 2 |
| dimension_deg_required_count | 0 | 0 | 2 |
| dimension_mm_measured_count | 0 | 0 | 2 |
| dimension_mm_required_count | 0 | 0 | 2 |
| dimension_ratio_measured_count | 0 | 0 | 2 |
| dimension_ratio_required_count | 0 | 0 | 2 |
| edge_chamfer_brep_mm2 | не считалось | не считалось | 0 |
| edge_chamfer_mesh_proxy_mm2 | 0.183077 | 0.183077 | 2 |
| facts.area_mm2_relative_error | 5.11123e-13 | 5.11123e-13 | 2 |
| facts.center_error_mm | 0 | 0 | 2 |
| facts.dimension_sets.diameter.matched | 9.5 | 9.5 | 2 |
| facts.dimension_sets.diameter.precision | 1 | 1 | 2 |
| facts.dimension_sets.diameter.prediction_count | 9.5 | 9.5 | 2 |
| facts.dimension_sets.diameter.recall | 1 | 1 | 2 |
| facts.dimension_sets.diameter.reference_count | 9.5 | 9.5 | 2 |
| facts.dimension_sets.linear.matched | 455.5 | 455.5 | 2 |
| facts.dimension_sets.linear.precision | 1 | 1 | 2 |
| facts.dimension_sets.linear.prediction_count | 455.5 | 455.5 | 2 |
| facts.dimension_sets.linear.recall | 1 | 1 | 2 |
| facts.dimension_sets.linear.reference_count | 455.5 | 455.5 | 2 |
| facts.dimension_sets.radius.matched | 1 | 1 | 2 |
| facts.dimension_sets.radius.precision | 1 | 1 | 2 |
| facts.dimension_sets.radius.prediction_count | 1 | 1 | 2 |
| facts.dimension_sets.radius.recall | 1 | 1 | 2 |
| facts.dimension_sets.radius.reference_count | 1 | 1 | 2 |
| facts.dimension_sets.thread.matched | 0 | 0 | 2 |
| facts.dimension_sets.thread.precision | 1 | 1 | 2 |
| facts.dimension_sets.thread.prediction_count | 0 | 0 | 2 |
| facts.dimension_sets.thread.recall | 1 | 1 | 2 |
| facts.dimension_sets.thread.reference_count | 0 | 0 | 2 |
| facts.mass_g_relative_error | не считалось | не считалось | 0 |
| facts.principal_moments_relative_error | не считалось | не считалось | 0 |
| facts.topology_delta.edges | 0 | 0 | 2 |
| facts.topology_delta.faces | 0 | 0 | 2 |
| facts.topology_delta.vertices | 0 | 0 | 2 |
| facts.volume_mm3_relative_error | 2.69004e-14 | 2.69004e-14 | 2 |
| flux_enclosure_error_mm2 | 1.32873e-12 | 1.32873e-12 | 2 |
| flux_vector_norm_mm2 | 1.15461e-12 | 1.15461e-12 | 2 |
| g_csr_project | не считалось | не считалось | 0 |
| normal_consistency | 0.999924 | 0.999924 | 2 |
| normal_consistency_oriented | 0.999924 | 0.999924 | 2 |
| point_to_point.chamfer_l1_mm | 1.11068 | 1.11068 | 2 |
| point_to_point.chamfer_l2_mm2 | 0.875366 | 0.875366 | 2 |
| point_to_point.hausdorff95_mm | 1.08675 | 1.08675 | 2 |
| point_to_point.hausdorff_mm | 1.9237 | 1.9237 | 2 |
| point_to_surface.chamfer_l1_mm | 0.000895415 | 0.000895415 | 2 |
| point_to_surface.chamfer_l2_mm2 | 3.98962e-06 | 3.98962e-06 | 2 |
| point_to_surface.hausdorff95_mm | 0.00278782 | 0.00278782 | 2 |
| point_to_surface.hausdorff_mm | 0.00374345 | 0.00374345 | 2 |
| precision | 1 | 1 | 2 |
| recall | 1 | 1 | 2 |
| segment_error | 0 | 0 | 2 |
| self_intersection_ratio | 0 | 0 | 2 |
| siou | 1 | 1 | 2 |
| siou_cadbench_formula_p2p | 0.934174 | 0.934174 | 2 |
| solid.extra_material_mm3 | 7.27596e-12 | 7.27596e-12 | 2 |
| solid.missing_material_mm3 | 2.59206e-10 | 2.59206e-10 | 2 |
| solid.symmetric_difference_mm3 | 2.67391e-10 | 2.67391e-10 | 2 |
| specification_compliance | не считалось | не считалось | 0 |

## Причины пропусков

- KF0017: Нужны выборки B-rep-кромок обеих моделей.
- KF0017: Не указан материал в снимках измерений; сопоставимость масс не подтверждена.
- KF0017: В evaluation.json не задан список требований чертежа (requirements).
- KF0027: Нужны выборки B-rep-кромок обеих моделей.
- KF0027: Не указан материал в снимках измерений; сопоставимость масс не подтверждена.
- KF0027: В evaluation.json не задан список требований чертежа (requirements).

VSR: 1; IR: 0.

Подробности, причины пропусков, исходная поза и преобразования — в `metrics.json`.
Запросы к КОМПАС — в `raw/solid_requests.json`; карты расстояний — в `raw/<id>/deviations.json`.
Аналитический стенд проверяет вычислитель и не является оценкой реконструкции CAD.
