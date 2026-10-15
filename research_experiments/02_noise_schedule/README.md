# 02_noise_schedule: Уровни шума

Сравнить full, single-level и короткие расписания при одинаковом числе основных score-шагов.

Все основные расписания получают 45 шагов плюс 10 final. У каждого одинаковая сетка длины. Scaled-full — отдельная координатная проверка, её нельзя смешивать с выбором количества уровней.

Для MNIST используйте `02_noise_schedule.mnist.pilot.json`. После полного development-сравнения зафиксируйте длины и заранее выбранные контрасты в confirmation-конфиге.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/02_noise_schedule/run.py --dry-run --max-jobs 2
python research_experiments/02_noise_schedule/run.py
python research_experiments/02_noise_schedule/run.py --resume
```

Pilot: `configs/02_noise_schedule.pilot.json`, 48 jobs. Варианты: `full`, `scaled_full`, `short`, `single0`, `single100`, `single300`, `single500`, `three`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/02_noise_schedule.confirmation.json
python research_experiments/02_noise_schedule/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/02_noise_schedule/analyze.py \
  research_experiments/results/02_noise_schedule/development \
  --output research_experiments/results/analysis_02_noise_schedule_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).
