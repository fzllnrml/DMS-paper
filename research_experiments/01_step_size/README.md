# 01_step_size: Длина шага

Проверить, объясняет ли фактическая длина движение и выигрыш.

Основной exact sweep использует L2; дополнительные `01_step_size.mnist.*.json` используют RMS. Совпадающая геометрия RMS/L2 проверяется тестами, дублирующих sweeps для доказательства тождества нет.

Включённый отдельно `01_step_size.basins.pilot.json` вычисляет численные бассейны после поиска. Ошибка сходимости помечается -1, а не заменяется ближайшей модой.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/01_step_size/run.py --dry-run --max-jobs 2
python research_experiments/01_step_size/run.py
python research_experiments/01_step_size/run.py --resume
```

Pilot: `configs/01_step_size.pilot.json`, 16 jobs. Варианты: `l2_length`, `smaller_more_steps`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/01_step_size.confirmation.json
python research_experiments/01_step_size/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/01_step_size/analyze.py \
  research_experiments/results/01_step_size/development \
  --output research_experiments/results/analysis_01_step_size_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
