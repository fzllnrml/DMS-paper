# 06_synthetic_distributions: Синтетический стенд

Проверить расстояния, перекрытие, редкость, анизотропию, много мод и изменение их числа.

Новые семейства фиксируют относительную сигнальную разделённость при изменении d; legacy_random намеренно воспроизводит старую box-геометрию. Все новые семейства сначала используют exact score.

Проверяйте `reference.json`. Полное одномерное сведение реализовано для <=2 компонент с общей ковариацией. Для остальных случаев `complete=false`; recall условен на найденном численном эталоне.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/06_synthetic_distributions/run.py --dry-run --max-jobs 2
python research_experiments/06_synthetic_distributions/run.py
python research_experiments/06_synthetic_distributions/run.py --resume
```

Pilot: `configs/06_synthetic_distributions.pilot.json`, 30 jobs. Варианты: `anisotropic`, `basins`, `bifurcation`, `historical_geometry`, `many`, `moderate`, `overlap`, `rare`, `separated`, `uneven`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/06_synthetic_distributions.confirmation.json
python research_experiments/06_synthetic_distributions/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/06_synthetic_distributions/analyze.py \
  research_experiments/results/06_synthetic_distributions/development \
  --output research_experiments/results/analysis_06_synthetic_distributions_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
