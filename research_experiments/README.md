# Дополнительные эксперименты DMS

Новая независимая директория для проверки длины шага, многоуровневости,
Split, Merge, покрытия и стоимости. Существующие файлы проекта не изменяются.
Обучения и автоматического скачивания данных нет. Все научные запуски выполняет автор.

Основа интеграции: `fzllnrml/DMS-paper`, коммит
`38499930c73a67c712d500d439fd74b682ef9651` от 15.10.2026.
Подробности: [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md),
[EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md), [VALIDATION.md](VALIDATION.md).

## 1. Применить патч

Из корня вашей копии DMS-paper, когда директории `research_experiments/` ещё нет:

```bash
git apply --check ~/Downloads/DMS_research_experiments.patch
git apply ~/Downloads/DMS_research_experiments.patch
git status --short
```

Вместо патча можно распаковать архив в корень проекта. Используйте один способ.
Не применяйте поверх другой версии этой директории. Никаких push/merge эти команды не выполняют.

## 2. Окружение

Python 3.12 соответствует README исходного проекта. Можно использовать ваш существующий venv.
Если его нет:

```bash
python3 -m venv research_experiments/.venv
source research_experiments/.venv/bin/activate
python -m pip install -r research_experiments/requirements-core.txt
```

Для исторических алгоритмов, Split-ядра, learned score и MNIST нужны зависимости проекта:

```bash
python -m pip install -r requirements-mnist-experiments.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Новый код поддерживает `cpu`, `mps`, `auto`; `auto` выбирает MPS при доступности,
иначе CPU. CUDA не требуется. Exact score вычисляется NumPy на CPU.
CPU fallback операций MPS и выбор полностью CPU — разные варианты:
для второго добавьте `--device cpu` **до** фиксации протокола.

## 3. Первые безопасные проверки

```bash
python -m unittest research_experiments.tests.test_core -q
python -m research_experiments.run doctor \
  --config research_experiments/configs/01_step_size.pilot.json --max-jobs 1
python -m research_experiments.run run \
  --config research_experiments/configs/01_step_size.pilot.json --dry-run --max-jobs 2
```

`doctor` и `--dry-run` читают конфигурацию и проверяют наличие входных путей.
Они не импортируют torch, не загружают веса, не вычисляют score и не создают результаты.
Наличие файла ещё не доказывает совместимость его содержимого: это проверяется при обычном запуске.

После просмотра плана первый небольшой **научный** запуск, который выполняете вы:

```bash
python -m research_experiments.run run \
  --config research_experiments/configs/01_step_size.pilot.json --max-jobs 2
```

Это exact-score поиск, обучение и checkpoints ему не нужны. `--max-jobs 2`
берёт первые два jobs в детерминированном плане, а не весь перебор длин шага.

## 4. Независимые блоки

Для каждого блока есть `run.py`, `analyze.py`, собственный README и JSON.
Общий CLI даёт тот же результат. Например:

```bash
python research_experiments/02_noise_schedule/run.py --dry-run --max-jobs 2
python research_experiments/02_noise_schedule/run.py
python research_experiments/02_noise_schedule/run.py --resume
```

| Блок | Что проверяется | Нужны веса |
|---|---|---|
| [00](00_retrospective/README.md) | Старые координаты/метрики и ограничения архива | Нет |
| [01](01_step_size/README.md) | Длина шага, clipping, колебания; отдельная диагностика бассейнов | Нет для exact; нужны для MNIST |
| [02](02_noise_schedule/README.md) | Полное, одноуровневое и сокращённые расписания; масштабированные координаты | Нет для exact |
| [03](03_baselines/README.md) | DDIM, кластеризация, независимый ascent, исторические ядра, b0/b10 | Torch для старых классов; exact не требует весов |
| [04](04_merge/README.md) | Время Merge, радиус, mean/medoid, стабилизация и перераспределение бюджета | Нет для exact |
| [05](05_split/README.md) | Старые v2/v3f1 Split-ядра против независимых стартов | Torch для импорта исходных классов; веса не нужны |
| [06](06_synthetic_distributions/README.md) | Близость, перекрытие, редкость, геометрия, размерность, бифуркации | Нет |
| [07](07_budget_pareto/README.md) | P, K, sampling/refinement, бюджет и Парето | Нет для exact |
| [08](08_exact_vs_learned/README.md) | Ошибка learned score на посещённых траекториях, общие learned-DDIM старты | Да, синтетические из репозитория |
| [09](09_mnist_diagnostics/README.md) | Картинки, классы, kNN, support, разнообразие и движение | Да, локальные MNIST и evaluator |

У 01 и 02 есть дополнительные конфиги `*.mnist.pilot.json` и `*.mnist.confirmation.json`.
У 01 есть небольшой `01_step_size.basins.pilot.json` с численной проверкой бассейнов
чистого exact-поля. Это дополнительная оптимизация после поиска, её стоимость записывается отдельно.

Пример последовательного плана двух блоков:

```bash
python -m research_experiments.batch --configs \
  research_experiments/configs/01_step_size.pilot.json \
  research_experiments/configs/02_noise_schedule.pilot.json \
  --dry-run --max-jobs 2
```

Уберите `--dry-run`, когда решите выполнить эти jobs. Для продолжения добавьте `--resume`.
Каждый job сохраняется отдельно. Не нужно повторять завершённые jobs после ошибки следующего.

## 5. Подключить существующие модели и данные

Синтетические checkpoints 10/30/50D уже есть в Git. Блок 08 использует
`checkpoints/dim_{dim}/model_dim{dim}_K{components}_dseed{distribution_seed}.pth`.
Он воспроизводит генератор **legacy_random** с тем же порядком RNG-вызовов.
Подставлять эти модели для новых близких/анизотропных смесей запрещено проверкой совместимости.

В Git нет этих четырёх MNIST-файлов:

```text
runs/mnist/checkpoints/best.pt
runs/mnist_seed43/checkpoints/best.pt
runs/mnist_seed44/checkpoints/best.pt
runs/mnist_evaluator/best.pt
```

При наличии их в старой локальной папке подключите родительский `runs`, без копирования:

```bash
python -m research_experiments.run doctor \
  --config research_experiments/configs/09_mnist_diagnostics.pilot.json \
  --mnist-root "$HOME/Desktop/diffusion-mode-search/runs" \
  --data-root "$HOME/Desktop/diffusion-mode-search/data/mnist" --max-jobs 1
```

Те же аргументы передавайте командам `run` и `lock`. Если файлы уже лежат в `DMS-paper/runs`
и `DMS-paper/data/mnist`, переопределения не нужны. Для переноса синтетических весов есть
`--checkpoint-root`, для старых результатов — `--archive-root`.
Невалидные/отсутствующие входы дают ошибку; никакого обучения взамен не запускается.
В MNIST проверяются записанный training seed и совпадение train/validation индексов evaluator и diffusion.

### Повторно использовать старые DDIM-старты

Новые jobs автоматически используют общий проверяемый NPZ-кэш стартов.
Он не уменьшает **приписанный методу** NFE: сохраняются intrinsic и фактически выполненные вызовы.
Первые строки NumPy-шума совпадают при разных P и одинаковом seed.
Это новая последовательность RNG, не побитовый replay прежнего torch RNG.

Старые `src/mnist_experiments/run.py` сохраняли `метод/arrays.pt` и `метод/summary.json`.
Импорт не выполняет inference:

```bash
python -m research_experiments.import_starts \
  /полный/путь/к/старому/методу/arrays.pt \
  --output research_experiments/results/imported/model42_seed0.npz
```

В копии конфигурации задайте `initialization.saved_path`:
`research_experiments/results/imported/model{model_seed}_seed{seed}.npz`.
Проверяются SHA checkpoint, seed, число DDIM-шагов, массив и стоимость.
Используется только допустимый префикс P. Старые старты не становятся независимым confirmation.

## 6. Разработка и confirmation

`*.pilot.json` — development. `*.confirmation.json` — отдельные запланированные сравнения.
Для exact-синтетики confirmation использует distribution seeds 101–105 и search seeds 1000–1004.
Для MNIST — существующие модели 43/44 и новые search seeds 100–104.
Для блока 08 новые совместимые модели не обучаются: остаются прежние распределения/веса,
меняются поисковые seeds. Это условная проверка на фиксированных моделях.

Полные confirmation-конфиги широкие: в блоке 07 по умолчанию **11 900 jobs**.
Сначала просмотрите план. Можно заранее сократить сетку в копии JSON и задать ей другой
`output_root` внутри `research_experiments/results/`. Фиксируйте конечный протокол до просмотра его результатов.
Научные метрики ground truth используются только после завершения поиска; автоматического подбора по ним нет.

Пример фиксации одного блока (команда напечатает полный путь протокола):

```bash
python -m research_experiments.run lock \
  --config research_experiments/configs/01_step_size.confirmation.json \
  --development-root research_experiments/results/01_step_size/development
```

Далее подставьте напечатанный путь:

```bash
python -m research_experiments.run run \
  --config research_experiments/configs/01_step_size.confirmation.json \
  --lock research_experiments/results/protocols/НАПЕЧАТАННЫЙ_SHA256.json
```

Для продолжения — та же команда с `--resume`. Код, устройство, конфигурация, пути и SHA входов
должны совпасть с lock. `--job 'label*'` выбирает заранее заданный вариант; фиксация всё равно
охватывает весь переданный конфиг. Не отбирайте удачные jobs после просмотра confirmation.
Отдельный confirmation для 00 отсутствует: старые данные не являются новой независимой проверкой.

## 7. Сохранение, ошибки и продолжение

Все записи ограничены `research_experiments/results/`, включая caches, протоколы и анализ.
Защита учитывает symlinks и `..`. В новых исходниках и конфигурациях ничего не записывается при запуске.
Job содержит `metadata.json`, `resolved_config.json`, `initialization.json`, `candidates.npz`,
`trajectory.npz/json`, `movement.json`, `stages.json`, `split_events.json`, `summary.json`
и контрольные суммы в `DONE.json`. Синтетика добавляет распределение/эталон; MNIST — logits,
признаки, изображения, соседей и показатели каждого выбранного этапа.

Файл `RUNNING` исключает второго писателя. После штатного исключения он удаляется.
После принудительного завершения процесса сначала убедитесь, что он остановлен:

```bash
python -m research_experiments.run unlock research_experiments/results/БЛОК/ФАЗА/JOB_ID
```

Команда отказывается снимать lock живого PID. `--resume` проверяет контрольные суммы и контекст.
Завершённые jobs пропускаются; незавершённый job повторяется с начала с тем же seed и кэшем стартов.
Состояние внутри старого исторического `find_modes` не сериализуется.
Изменив код/параметры/входы, используйте новый `output_root`; существующие результаты не перезаписываются.

## 8. Анализ без моделей

После ваших запусков:

```bash
python -m research_experiments.analysis.analyze \
  research_experiments/results/02_noise_schedule/development \
  --output research_experiments/results/analysis_02_v1 \
  --contrast full:single300 --bootstrap 10000
```

Появятся `runs.csv`, `aggregates.csv`, `paired_contrasts.csv`,
`pareto.csv`, `pareto_aggregates.csv`, PNG и описание анализа.
Радиусы, семейства, размерности, exact/learned и фазы не смешиваются.
Настройки длины/числа шагов не усредняются как повторные seeds: у каждой есть `setting`.
Парный анализ требует одинаковых хэшей стартов. Он не соединяет разные P или разную инициализацию.
Интервалы учитывают смесь/модель, затем search seed; частицы не являются повторами.
Для всех запрошенных контрастов/настроек/радиусов/метрик применяется поправка Bonferroni.
`extreme_tail_expected_draws` показывает, достаточно ли bootstrap-повторов для крайних квантилей.
При одной смеси/модели интервалы условны на ней; при одном запуске интервал не выдумывается.

Дополнительные графики сохранённого job:

```bash
python -m research_experiments.analysis.trajectory_plots \
  research_experiments/results/БЛОК/ФАЗА/JOB_ID \
  --output research_experiments/results/trajectory_JOB_ID
python -m research_experiments.analysis.mnist_plots \
  research_experiments/results/09_mnist_diagnostics/development/JOB_ID \
  --output research_experiments/results/images_JOB_ID
```

Сетки используют заранее заданные исходные индексы, а не красивость.
При утрате однозначного происхождения ячейка помечается, изображение другого предка не подставляется.

## 9. Как читать результаты

* RMS и fixed-L2 с согласованной длиной — один шаг. Это проверяется unit-тестом,
  а не оплачивается двумя одинаковыми научными sweeps.
* Без Split/Merge одинаковый multi-level ascent — тот же независимый многоточечный подъём.
  Для эквивалентности есть тест; искусственного конкурирующего метода не создано.
* В пределах радиуса conventional precision может засчитывать дубликаты. Основной контроль
  компактности — maximum-cardinality one-to-one matching с минимальной суммой расстояний среди максимумов.
* Для общей ковариации и не более двух компонент полный набор строгих мод восстанавливается
  одномерным сведением; вырожденные случаи помечаются. Для прочих смесей multistart даёт
  **неполный численный эталон**. Recall/F1 тогда условны на найденном эталоне.
* Mean Merge способен переместить представителя и повысить либо снизить радиусный recall.
  Геометрическая близость сама по себе не доказывает принадлежность одной моде.
* Перераспределение бюджета — явно заданная эвристика. Hard cap действует отдельно в единицах
  поля поиска. Neural NFE и analytic evaluations сохраняются раздельно и не приравниваются по цене.
* При нехватке бюджета обычный ascent сохраняет последнее завершённое обновление;
  неполный Split не применяется. Исторический атомарный метод при исчерпании cap даёт ошибку,
  поэтому его предустановленные сравнения без cap оцениваются по фактической стоимости.
* Если после Merge осталось меньше заданного K, точки не дублируются для заполнения.
  Проверяйте `requested_k` и `k_constraint_met` перед сравнением с фиксированным K.
* kNN, support и классы MNIST характеризуют поддержку и семантику, а не истинную модальность.
  Pilot использует сокращённую reference-базу; его числа нельзя подставлять в таблицы статьи.

Из дорогих операций заранее учитывайте полный DDPM в b0/b10 (1000 вызовов на объект),
полные FD-гессианы (2d score-вызовов на проверку), квадратичную кластеризацию,
многостартовую проверку эталонов, анализ бассейнов и kNN по 55 000 MNIST-изображений.
Время и память зависят от реализации/батчинга, не только от NFE.
`search_seconds` включает фактическую инициализацию/чтение кэша и поиск,
но не загрузку весов и внешнюю оценку. Для сопоставления полного wall time
отключите повторное использование стартов у всех методов: `initialization.cache=false`
в копии конфигурации. При включённом кэше время условно на его состоянии;
`initialization_reused` сохраняется в каждой строке.
