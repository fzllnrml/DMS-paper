# 08_exact_vs_learned: Точный и обученный score

Измерить ошибки поля на действительно посещённых координатах при одинаковых стартах.

Требуются существующие `checkpoints/dim_{dim}/model_dim{dim}_K{components}_dseed{distribution_seed}.pth`; новые семейства с ними не совместимы. Старты в обеих ветвях одинаковые learned-DDIM, отдельно начисляются neural NFE.

Exact использует alpha/beta массива загруженной модели. Диагностика обоих полей проводится после получения кандидатов и начисляется отдельно. Никаких новых обучений и новых независимых training seeds нет.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/08_exact_vs_learned/run.py --dry-run --max-jobs 2
python research_experiments/08_exact_vs_learned/run.py
python research_experiments/08_exact_vs_learned/run.py --resume
```

Pilot: `configs/08_exact_vs_learned.pilot.json`, 4 jobs. Варианты: `exact`, `learned`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/08_exact_vs_learned.confirmation.json
python research_experiments/08_exact_vs_learned/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/08_exact_vs_learned/analyze.py \
  research_experiments/results/08_exact_vs_learned/development \
  --output research_experiments/results/analysis_08_exact_vs_learned_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
