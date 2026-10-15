# 09_mnist_diagnostics: MNIST

Сохранить фиксированные примеры, смены классов, kNN/support, разнообразие и движение.

Требуются локальные MNIST U-Net, evaluator и датасет. Pilot: model42, 2048 train references и 128 validation; confirmation: model43/44, полный train и 1024 validation. Это разные точности оценки.

Второй независимый classifier того же формата подключается через mnist.second_evaluator_checkpoint, если он уже существует. Классы и признаки используются только после поиска; новой тренировки нет.

## Запуск

Все команды выполняются из корня DMS-paper. Установка: [общий README](../README.md#2-окружение).
На exact score для этого блока обучение не запускается. Все пути можно проверить заранее.

```bash
python research_experiments/09_mnist_diagnostics/run.py --dry-run --max-jobs 2
python research_experiments/09_mnist_diagnostics/run.py
python research_experiments/09_mnist_diagnostics/run.py --resume
```

Pilot: `configs/09_mnist_diagnostics.pilot.json`, 9 jobs. Варианты: `ddim`, `final_medoid`, `intermediate`, `large_step`, `multi`, `short`, `single100`, `single300`, `small_step`.

Confirmation фиксируется до запуска:

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/09_mnist_diagnostics.confirmation.json
python research_experiments/09_mnist_diagnostics/run.py --confirmation \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Те же локальные path overrides и device передаются и `lock`, и `run`. Добавьте `--resume` для продолжения.

## Анализ

```bash
python research_experiments/09_mnist_diagnostics/analyze.py \
  research_experiments/results/09_mnist_diagnostics/development \
  --output research_experiments/results/analysis_09_mnist_diagnostics_v1
```

Новый каталог анализа не должен существовать. Для парных сравнений добавьте `--contrast label_a:label_b`.

## Выходы и трактовка

Сырые `candidates.npz`, отдельные `trajectory.npz/json`, движения, seeds, параметры и стоимость сохраняются рядом с `summary.json` и `DONE.json`. Анализ создаёт CSV, интервалы, отдельные Парето-оси и графики из сохранённых файлов, без повторного inference. Для 00 основным сырым отчётом является `retrospective.json`; при отсутствии массивов экспортируются только архивные числа с соответствующей пометкой.

Гипотезы, положительные и отрицательные исходы, контролируемые параметры и пределы выводов: [EXPERIMENT_MATRIX](../EXPERIMENT_MATRIX.md). Фактические зависимости: [REPOSITORY_AUDIT](../REPOSITORY_AUDIT.md).

Пример подключения локальных входов:

```bash
python research_experiments/09_mnist_diagnostics/run.py --dry-run --max-jobs 1 \
  --mnist-root "$HOME/Desktop/diffusion-mode-search/runs" \
  --data-root "$HOME/Desktop/diffusion-mode-search/data/mnist"
```

Для сохранённых картинок используйте `python -m research_experiments.analysis.mnist_plots JOB --output НОВЫЙ_КАТАЛОГ_В_results`.
