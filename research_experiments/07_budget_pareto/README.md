# 07_budget_pareto: Бюджет и Парето

Сравнить распределения бюджета sampling/refinement, разные P и ограничения K.

Pilot — ограниченный набор; полная confirmation-сетка содержит 11 900 jobs. Не запускайте её целиком без просмотра и предварительного сужения. Никакой weighted composite score не вводится.

Проверяйте unused_budget, requested_k и k_constraint_met. Дефицит кандидатов после Merge не заполняется копиями. Для измерения полного времени без кэша задайте initialization.cache=false у всех методов.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/07_budget_pareto/run.py --dry-run --max-jobs 2
python research_experiments/07_budget_pareto/run.py
python research_experiments/07_budget_pareto/run.py --resume
```

Pilot: `configs/07_budget_pareto.pilot.json`, 17 jobs. Варианты: `matched_sampling_20`, `matched_sampling_50`, `merge`, `refinement`, `restarts`, `sampling_P16`, `sampling_P32`, `sampling_P64`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/07_budget_pareto.confirmation.json
python research_experiments/07_budget_pareto/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/07_budget_pareto/analyze.py \
  research_experiments/results/07_budget_pareto/development \
  --output research_experiments/results/analysis_07_budget_pareto_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
