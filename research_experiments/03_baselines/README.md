# 03_baselines: Сильные baseline

Определить вклад популяционных операций относительно простых конкурентов.

Реализованы DDIM/clustering, independent ascent, matched-pool medoids, KDE mean shift, facility selection, исторические v1/v2/v3f1/v3f2/v3f3 и b0/b10 с полным DDPM.

Для импорта старых классов нужен torch даже на exact score. Старые smart-start процедуры заменены контролируемыми внешними стартами; v3f3 получает свой корректный warm endpoint t=800. DDPM b0/b10 значительно дороже.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/03_baselines/run.py --dry-run --max-jobs 2
python research_experiments/03_baselines/run.py
python research_experiments/03_baselines/run.py --resume
```

Pilot: `configs/03_baselines.pilot.json`, 30 jobs. Варианты: `b0_ddpm`, `b10_ddpm`, `ddim`, `ddim_cluster`, `ddim_facility`, `ddim_mean_shift`, `historical_v1`, `historical_v2`, `historical_v3f1`, `historical_v3f2`, `historical_v3f3`, `independent`, `independent_final_cluster`, `matched_pool_medoid`, `single_level`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/03_baselines.confirmation.json
python research_experiments/03_baselines/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/03_baselines/analyze.py \
  research_experiments/results/03_baselines/development \
  --output research_experiments/results/analysis_03_baselines_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
