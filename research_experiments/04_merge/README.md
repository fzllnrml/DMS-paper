# 04_merge: Merge

Проверить compactness, потери мод, экономию и перераспределение вычислений.

Поддержаны none/final/every/periodic/noise-scaled/stable и direction gate, mean/medoid, complete/single/average linkage; grid использует complete. Reallocation: продолжение, новые старты, jitter наиболее изолированной точки.

Stable означает последний малый projected movement всех частиц. В нормированном ascent такой критерий может срабатывать редко — это наблюдаемый результат. Он не доказывает истинную локальную модальность.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/04_merge/run.py --dry-run --max-jobs 2
python research_experiments/04_merge/run.py
python research_experiments/04_merge/run.py --resume
```

Pilot: `configs/04_merge.pilot.json`, 34 jobs. Варианты: `aligned`, `final`, `intermediate_mean`, `intermediate_medoid`, `noise_radius`, `none`, `periodic`, `reallocate_continue`, `reallocate_jitter`, `reallocate_restarts`, `stable`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/04_merge.confirmation.json
python research_experiments/04_merge/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/04_merge/analyze.py \
  research_experiments/results/04_merge/development \
  --output research_experiments/results/analysis_04_merge_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
