# 05_split: Split

Сравнить локальные probes с дополнительными независимыми стартами.

Переиспользуются неизменённые NumPy-методы v2 и v3f1; есть axes и boundary-norm gate. Полный Hessian требует 2d score evaluations. Отдельного дешёвого HVP-алгоритма нет.

Итоговое покрытие сравнивайте на равном фактическом бюджете. `new_unique_reference_hits` относится к полной популяции сразу после события, не к причинному доказательству финального открытия мод.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/05_split/run.py --dry-run --max-jobs 2
python research_experiments/05_split/run.py
python research_experiments/05_split/run.py --resume
```

Pilot: `configs/05_split.pilot.json`, 16 jobs. Варианты: `boundary_kernel`, `coordinate_probes`, `historical_v2`, `historical_v3f1`, `kernel_v2`, `kernel_v3f1`, `more_independent`, `no_split`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/05_split.confirmation.json
python research_experiments/05_split/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/05_split/analyze.py \
  research_experiments/results/05_split/development \
  --output research_experiments/results/analysis_05_split_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
