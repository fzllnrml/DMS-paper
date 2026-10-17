# DMS: независимый экспериментальный комплекс v2

`research_experiments/` — сохранённый комплекс первой серии E0–E9.
`research_experiments_v2/` — независимое продолжение исследования. Старые
исходники, конфигурации, checkpoints и результаты не исправляются и не
перезаписываются. Патч добавляет только эту новую директорию.

Основа: `fzllnrml/DMS-paper`, `main`, commit
`10454bee79847a1246662fa5c280aba8b71b6801`. Проверенные математические функции
v1 импортируются без изменения; [точные зависимости и различия](docs/DEPENDENCIES_AND_CHANGES_RU.md).
V2 не является автономной заменой всего репозитория.

Все вычисления ниже выполняет пользователь локально. При подготовке патча
выполнялись только unit-тесты с маленькими искусственными массивами,
проверки конфигураций, dry-run и чтение присланных таблиц. Обучение,
checkpoint inference и научные серии не запускались.

## Сначала об архиве первой серии

Архива достаточно для аудита итоговых таблиц и разработки следующего этапа.
Он содержит 204 jobs и 789 строк метрик, а не 789 независимых экспериментов.
Первичных массивов и траекторий в нём нет. Пустой `paired_contrasts.csv`
согласуется с `contrasts: []` в `analysis.json`.

[Аудит достаточности и конкретные недостающие файлы](docs/V1_ARCHIVE_AUDIT_RU.md).
[Машиночитаемый запрос исходных данных](docs/v1_evidence_request.json) содержит
точные относительные пути всех 204 jobs: небольшой комплект метаданных для
всех jobs и приоритетные массивы для 69. Эти пробелы не блокируют запуск v2.

## Установка на MacBook

Все команды — из корня `DMS-paper`. Сначала применить отдельный патч:

```bash
git apply --check /path/to/DMS-paper_research_v2.patch
git apply /path/to/DMS-paper_research_v2.patch
python3 -m venv research_experiments_v2/.venv
source research_experiments_v2/.venv/bin/activate
python -m pip install -r research_experiments_v2/requirements.txt
export PYTHONDONTWRITEBYTECODE=1
```

Использовать Python 3.11 или 3.12 и совместимую пару torch/torchvision для
своего macOS. `requirements.txt` задаёт допустимые диапазоны, а реальные
версии всех зависимостей записываются в результаты и фиксируются lock-файлом.
Окружение и датасеты не входят в патч. Данные не скачиваются автоматически;
отсутствующий checkpoint никогда не приводит к обучению.

```bash
python -m research_experiments_v2 doctor
python -m research_experiments_v2 doctor --inspect-checkpoints --device auto
python -m unittest discover -s research_experiments_v2/tests -v
python -m research_experiments_v2 verify-isolation
```

Обычный `doctor` проверяет пакеты, файлы и доступность запрошенного MPS.
`--inspect-checkpoints` дополнительно читает metadata, проверяет model seed,
расписание и совпадение train/validation split с evaluator. Forward pass
при этой проверке не выполняется. Для полного Pilot требуются torch и
scikit-learn даже в exact-блоках: они импортируются оригинальными Split/
историческими адаптерами; нейросеть там не используется.

## Внешние данные и модели

По умолчанию используются существующие пути репозитория:

| Вход | Путь / параметр |
|---|---|
| Синтетические модели | `checkpoints/dim_10/`, `dim_30/`, `dim_50/`; `--checkpoint-root /external/checkpoints` |
| MNIST model 42 | `runs/mnist/checkpoints/best.pt` |
| MNIST models 43/44 | `runs/mnist_seed43/checkpoints/best.pt`, `runs/mnist_seed44/checkpoints/best.pt` |
| Замена всего `runs/` | `--mnist-root /external/runs` |
| Frozen evaluator | `runs/mnist_evaluator/best.pt` |
| MNIST IDX | `data/mnist/MNIST/raw/`; `--data-root /external/data/mnist` |
| Произвольные отдельные файлы | `--model-paths /external/model_paths.json` |

Образец карты: [external_paths.example.json](configs/external_paths.example.json).
Это JSON «путь из конфигурации → внешний путь». Можно перенаправить любую
модель и evaluator независимо от структуры каталогов. Все входы читаются,
контентные SHA256 записываются. Параметры путей повторить при `doctor`, запуске
Pilot и создании confirmation lock; запуск Confirmation возьмёт их из lock.

Для MNIST нужны локальные распакованные
`MNIST/raw/train-images-idx3-ubyte` и `train-labels-idx1-ubyte`.
Используется вся train-часть сохранённого split и 1024 validation-примера,
без test set. Уменьшённые reference sets не подменяют основной протокол.

## Dry-run: без вычислений

```bash
python -m research_experiments_v2 dry-run
python -m research_experiments_v2 run-pilot --dry-run
python -m research_experiments_v2 dry-run --domain exact
python -m research_experiments_v2 dry-run --manifest research_experiments_v2/configs/confirmation_all.json
```

Dry-run разворачивает конфигурации и показывает входы, объём jobs и сумму
лимитов score queries. Он не строит эталоны, не загружает модели и не оценивает
score. Сумма лимитов не является прогнозом времени на MacBook.

## Весь Pilot v2 одной командой

```bash
mkdir -p research_experiments_v2/results/logs
caffeinate -i python -u -m research_experiments_v2 run-pilot --device auto > research_experiments_v2/results/logs/pilot.log 2>&1
```

Перед первым job проверяются входы всей выбранной серии. Успешные jobs
сохраняются атомарно, с SHA256 всех артефактов и `DONE.json`. Повтор **той же
команды** проверяет и пропускает готовые jobs, повторяет незавершённые.
Ошибки записываются в `ERROR.json`; остальные jobs продолжаются. Итоговый
exit code ненулевой при ошибках. Для немедленной остановки есть `--fail-fast`.

После аварийного завершения процесса повторить команду с `--recover-stale`:
блокировка снимается только для отсутствующего PID на том же компьютере.
Живой процесс или блокировка другого хоста не перехватывается.

Можно выполнить блок отдельно:

```bash
python -m research_experiments_v2 run-pilot --domain exact
python -m research_experiments_v2 run-pilot --domain alignment
python -m research_experiments_v2 run-pilot --domain mnist --device auto
```

Для одного файла используется `--config research_experiments_v2/configs/01_exact_mechanisms.pilot.json`.
`--labels ...` выбирает точные метки. Частичная серия помечается неполной в
аналитике и не выдаётся за весь Pilot.

| Pilot-блок | Jobs |
|---|---:|
| Exact mechanisms: 8 семейств × 3 размерности × 3 распределения × 2 search seeds × 2 бюджета × 8 методов | 2304 |
| Exact cap | 324 |
| Merge/Split ablations | 540 |
| Шаг и расписание | 486 |
| P=4/16 при фиксированном бюджете | 288 |
| Оригинальные классы с общими внешними стартами | 144 |
| Exact/learned, существующие K=2 checkpoints | 72 |
| MNIST tune: model42, search0–4, P128 | 150 |
| MNIST фиксированный exploratory transfer: models43/44 | 80 |
| MNIST cap, включая DDIM50 P128 | 20 |
| **Всего** | **4408** |

Научные вопросы, гипотезы, альтернативные исходы и ограничения:
[экспериментальная матрица](docs/EXPERIMENT_MATRIX_RU.md).

## Анализ результатов

```bash
python -m research_experiments_v2 analyze --plots
```

В `results/analysis_pilot/` появятся:

- `runs.csv`, `aggregates.csv`: настройки не усредняются в одну метку;
- `paired_contrasts.csv`: заранее заданные пары, интервалы, причины исключения;
- `paired_differences.csv`: индивидуальные парные разности; невалидные помечены;
- `design_audit.csv`: отсутствующие jobs и неполные cohorts;
- `pareto.csv`, `pareto_aggregates.csv`, `pareto_status.csv`: фронты по отдельным
  задачам и по общему допустимому набору пар, без смешения размерностей,
  режимов K и аналитических/нейросетевых стоимостей;
- `mnist_guards.csv`, `hypothesis_decisions.csv`, `analysis.json`;
- `plots/`, `pareto_plots/`: PNG для последующего научного анализа.

Нет завершённых jobs — нет оценок и научных выводов. Файл контрастов всё равно
содержит явный статус отсутствия данных; пустая таблица не маскирует ошибку
протокола. [Правила статистики и оценивания](docs/PROTOCOL_RU.md).

У shared-start cache есть две стоимости: intrinsic queries списываются каждому
методу, физически выполненные вызовы записываются отдельно. Для сопоставления
wall time использовать отдельный `--run-root` и `--no-cache`; при анализе
повторить `--no-cache`. В основной аналитике кэшированные времена не
интерпретируются как честный выигрыш метода.

## Confirmation v2: отдельный запуск после фиксации

```bash
python -m research_experiments_v2 propose-selection
```

Просмотреть `results/protocols/selection_draft.json`: выбрать 4–8 MNIST
treatments с обязательными controls и оптимизированным single-level,
указать `mnist_roles.optimized_single`, гипотезы и причины оставшихся
tradeoffs. Заполнить `review_notes` и `reviewed: true`.
Численные кандидаты пересчитываются из model42 при создании lock;
подмена списка кандидатов не принимается.

```bash
python -m research_experiments_v2 lock-confirmation --selection research_experiments_v2/results/protocols/selection_draft.json --device auto
python -m research_experiments_v2 run-confirmation --lock research_experiments_v2/results/protocols/confirmation_lock.json --dry-run
```

Следующая команда — **только когда пользователь решит начать Confirmation**:

```bash
caffeinate -i python -u -m research_experiments_v2 run-confirmation --lock research_experiments_v2/results/protocols/confirmation_lock.json > research_experiments_v2/results/logs/confirmation.log 2>&1
python -m research_experiments_v2 analyze --root research_experiments_v2/results/confirmation --out research_experiments_v2/results/analysis_confirmation --lock research_experiments_v2/results/protocols/confirmation_lock.json --plots
```

Blueprint содержит 1740 потенциальных jobs; lock оставляет только выбранные
методы, обычно не более 1520 jobs (1260 exact matched, 180 exact cap, до 80
MNIST). Новые distribution seeds 101–105 и search seeds 1000+ не пересекаются
с Pilot. Сохранённые MNIST models43/44 не становятся новыми обучениями:
статистические выводы условны на этот фиксированный набор моделей.

Lock фиксирует код, версии пакетов, методы, гипотезы, контрасты и SHA256
входов. Смена кода или входных файлов после lock останавливает запуск.
`run-pilot` не позволяет обойти Confirmation-lock. Подробности:
[выбор настроек и Confirmation](docs/SELECTION_AND_CONFIRMATION_RU.md).

## Данные и изоляция

Все новые результаты, кэши, логи, экспорты v1 и lock-файлы пишутся только в
`research_experiments_v2/results/`. Выход за этот каталог, в том числе через
символическую ссылку в старые результаты, запрещён. `.gitignore` исключает
окружения, результаты, датасеты, модели, кэши и временные файлы.

Для контроля локальных старых результатов перед длительной серией можно
снять полный read-only snapshot файлов вне v2 (большие результаты требуют
времени на чтение):

```bash
python -m research_experiments_v2 snapshot-isolation
python -m research_experiments_v2 verify-isolation --baseline research_experiments_v2/results/audit/isolation_baseline.json
```

Старые CSV не загружаются анализатором v2. Единственное явное обращение к
первой серии — отдельные команды `audit-v1` / `export-v1-evidence` с маркировкой
`v1_archive_only` / `read_only_v1_export`.

[Технический отчёт](docs/TECHNICAL_REPORT_RU.md) ·
[проверка сохранности](docs/VERIFICATION_RU.md) ·
[ограничения checkpoint-совместимости](docs/CHECKPOINT_COMPATIBILITY_RU.md).
