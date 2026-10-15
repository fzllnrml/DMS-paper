# 00_retrospective: Повторный анализ

Сравнить исходные показатели с 1:1 matching, компактностью и стоимостью.

В Git есть сохранённые агрегаты, но нет координат validation и старых v1/v2/b0/b10. Без локальных NPZ отчёт перечислит недоступные методы и экспортирует архивные числа с исходным происхождением.

Сначала подключите `--archive-root /полный/путь/к/старым/результатам`. Validation NPZ распознаются автоматически. Для иных форматов создайте manifest по примеру ниже и задайте `retrospective.manifest` в копии JSON.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/00_retrospective/run.py --dry-run --max-jobs 2
python research_experiments/00_retrospective/run.py
python research_experiments/00_retrospective/run.py --resume
```

Pilot: `configs/00_retrospective.pilot.json`, 1 jobs. Варианты: `archive_inventory`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/00_retrospective.confirmation.json
python research_experiments/00_retrospective/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/00_retrospective/analyze.py \
  research_experiments/results/00_retrospective/development \
  --output research_experiments/results/analysis_00_retrospective_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).

## Manifest для старых внешних массивов

Это пример **схемы**, не существующий файл с научными результатами. Пути относительно manifest. NFE оставьте null, если он не сохранён; не восстанавливайте его предположением. Не отмечайте reference_complete=true без основания.

```json
{
  "runs": [
    {
      "path": "relative/to/manifest/raw_run.npz",
      "method": "v2",
      "dimension": 2,
      "candidate_key": "modes",
      "reference_key": "verified_modes",
      "reference_complete": false,
      "reference_status": "supply_your_verification_provenance",
      "seed": 0,
      "distribution_seed": 0,
      "neural_nfe": null,
      "analytic_evaluations": null
    }
  ]
}
```

Автоадаптер validation NPZ использует сохранённые параметры смеси и отдельно проверяет численные моды, не переименовывает `true_centers` в ground truth. Если сопутствующий cost-record отсутствует, NFE остаётся null.
