# Достаточность приложенного архива E0–E9

**Для описательного анализа и разработки следующего этапа архив достаточен.
Для причинного пересчёта отдельных механизмов — недостаточен.** Дополнительная
выгрузка нужна для проверки старых объяснений, но не является условием
разработки/запуска v2.

Проверены оба приложения: `DMS_all_pilot_analysis.zip` и
`DMS_pilot_scientific_review_RU.md`. Архив не используется как замена исходникам:
исходники взяты из актуального Git main commit
`10454bee79847a1246662fa5c280aba8b71b6801`.

SHA256 ZIP: `68a148a165fbdf1f2dee132f578b0ca915291b9194c66cef875b067e90710a98`.
SHA256 review: `ad85ce087fffa6c1836f50ba9d4a184d580b9218fd3df6aebc10fc4f16e5277d`.
ZIP: 220749 bytes, 10 файлов (плюс directory entries), 2622273 bytes
распакованных файлов. CRC проверен; повреждений не обнаружено.

## Что действительно приложено

| Содержимое | Что позволяет установить |
|---|---|
| `runs.csv`: 789 rows, 204 unique jobs | Сохранённые scalar metrics, радиусы, cost, P/K, частично provenance/pair hashes |
| `aggregates.csv`, `pareto.csv`, `pareto_aggregates.csv` | Старые сводки; labels могут объединять разные settings |
| `archived_per_run.csv`: 575 rows, `archived_aggregates.csv` | Исторические записи; не 575 новых независимых запусков |
| `paired_contrasts.csv`: **0 bytes** | Контрасты не были выведены |
| `analysis.json` | `contrasts: []`, bootstrap500; отсутствие запроса контрастов объясняет пустую таблицу |
| `archive_duplicates.json`: `[]`, night log | Служебная информация, не первичные траектории |

| Серия | Jobs |
|---|---:|
| E1 | 16 |
| E2 | 48 |
| E3 | 30 |
| E4 | 34 |
| E5 | 16 |
| E6 | 30 |
| E7 | 17 |
| E8 | 4 |
| E9 | 9 |
| **Всего** | **204** |

195 synthetic jobs: 191 в 2D и 4 в 10D (E8); ни одной новой 30D/50D задачи
в `runs.csv`. MNIST — 9 jobs, model42/search0/P16. Несколько radii одного
job не являются независимыми повторами. По архиву нельзя доказать точную
версию файлов, породивших результаты: для этого нужна per-job metadata.

## Какие выводы уже допустимы

- В E3 independent + final clustering и single-level дают 1:1 F1=1 и два
  выхода при 1200 аналитических queries в двух search seeds одной простой
  смеси. Это сильный baseline в указанной постановке.
- Historical v3f2 в E3 имеет средний recall .75 при меньшем среднем расходе
  515; DDIM — 320 против 1200 ascent. Это **не** matched-query superiority.
- В E5 no_split уже достигает recall=1. Дополнительный recall Split на этих
  данных не идентифицируется, хотя стоимость/компактность различаются.
- В E6 rare не находит редкую моду в обоих seeds. В 10 из 30 E6 jobs reference
  incomplete; два bifurcation jobs имеют вообще ноль эталонных мод.
- E7 Merge с requested K=8 возвращает два кандидата и false constraint flag.
  Такой результат нельзя включать в строгое K=8 сравнение.
- В E9 large_step лучше multi по обеим kNN distances при одинаковых 1680 NFE
  и 16 выходах, но это одна модель и один search seed. Intermediate Merge
  возвращает один объект/класс; его меньший kNN не доказывает coverage.
- E8 имеет одинаковые initial hashes по парным seeds, но incomplete reference
  во всех четырёх jobs. Gap нельзя целиком приписывать обучению без проверки
  checkpoint, единиц, schedule и траекторий.

Здесь не повторяются pooled means по разным настройкам E4 как причинное
сравнение конкретных радиусов. Review полезен как независимая интерпретация
таблиц; первичными доказательствами траекторий он не является.

## Недостающие файлы и их назначение

Базовый относительный путь:
`research_experiments/results/<suite>/development/<job_id>/`.

| Файл | Зачем нужен |
|---|---|
| `resolved_config.json` | Точные шаги, уровни, радиусы, Split thresholds, budget/output rule; отделение нескольких settings под одной label |
| `metadata.json`, `implementation.json`, `DONE.json` | Source/input hashes, версии, реальные historical constructor arguments, проверка целостности |
| `summary.json`, `initialization.json` | Полные cost components, время/кэш, происхождение и стоимость starts, constraints |
| `candidates.npz` | Начальные/все начальные/final arrays; независимый пересчёт one-to-one matching, дубликатов, displacement |
| `distribution.json`, `reference.npz`, `reference.json` | Реальная плотность, найденные моды и статус полноты; проверка ошибочного denominator |
| `trajectory.npz`, `trajectory.json` | Координаты на этапах, timestep, ancestry; восстановление направления и геометрии движения |
| `movement.json`, `stages.json` | Фактические шаги, clipping, population changes и расписание интервенций |
| `split_events.json`, `split_diagnostics.json` | Родители/дети, accepted/capped Split, цена и дополнительные hits |
| `alignment.json` (особенно E8) | Same-state exact/learned errors и basin diagnostics, уже посчитанные в первой серии |
| `mnist.npz`, `mnist.json` (E9) | Images/features/logits, neighbor IDs, распределения по стадиям и class changes; проверка коллапса и train-close artifacts |

Для повторной оценки MNIST нужны также те же локальные dataset split и frozen
evaluator/checkpoint hashes. Передавать сами MNIST/.venv или все модели в
архив первичных результатов не требуется: их можно использовать локально.

## Конкретные jobs для узкой первичной выгрузки

Все следующие пути находятся внутри указанного `suite/development/`:

| Серия / механизм | Точные job directories |
|---|---|
| E3 сильный independent baseline | `independent_final_cluster-9d9b8bdcf16c8d55`, `independent_final_cluster-1143e3769ee015af` |
| E3 потеря моды historical v3f2 | `historical_v3f2-62b30e5dba7f985c`, `historical_v3f2-52015d5d16411079` |
| E3 высокая стоимость historical v2 | `historical_v2-f203dcaea0c33724`, `historical_v2-d02f1adfc066d7d6` |
| E4 medoid collapse (один setting) | `intermediate_medoid-19d56c11f915777f`, `intermediate_medoid-a38b994e4354a288` |
| E4 reallocation | `reallocate_restarts-6c8905c9898b9006`, `reallocate_restarts-9bea62aaf7010da7` |
| E5 no_split vs original local kernel | `no_split-b688497d5b527520`, `no_split-0e705448d7f19bbe`, `kernel_v2-e8b76b58e0c6b7aa`, `kernel_v2-4773a7eeeef67215` |
| E6 rare | `rare-5c4dd66e663aeb7f`, `rare-f1c7d13f43ef7c77` |
| E6 пустой denominator | `bifurcation-7c092ebc26423c7b`, `bifurcation-eafa6bcf53ebc936` |
| E8 exact | `exact-a2362da68abe6f71`, `exact-bd442a878f49d766` |
| E8 learned | `learned-c033fb9d6a791469`, `learned-429446e532be3175` |
| E9 shared controls | `ddim-eb00e16106b18910`, `multi-6207f9d677046549`, `large_step-7620267077a64222` |
| E9 collapse / single / short | `intermediate-d56b89e05cf599a8`, `single300-2265c4c0f68084e1`, `single100-b498b9bdcc35490e`, `short-4dbce97ec595abb0` |
| E9 output/step controls | `final_medoid-413580fc178af3dc`, `small_step-962c078d5ba0545d` |

Полный список с точными suite, setting, seed и filenames находится в
`v1_evidence_request.json`: 204 metadata-комплекта, из них 69 с raw priority.
Это явный запрос доказательств первой серии, а не список новых jobs v2.

## Read-only команды

```bash
python -m research_experiments_v2 audit-v1 --archive /path/to/DMS_all_pilot_analysis.zip
python -m research_experiments_v2 export-v1-evidence --old-results-root /path/to/DMS-paper/research_experiments/results --dry-run
```

Минимальная тематическая выгрузка E8+E9 — 13 jobs:

```bash
python -m research_experiments_v2 export-v1-evidence --old-results-root /path/to/DMS-paper/research_experiments/results --suites 08_exact_vs_learned 09_mnist_diagnostics --tier raw_priority --dry-run
python -m research_experiments_v2 export-v1-evidence --old-results-root /path/to/DMS-paper/research_experiments/results --suites 08_exact_vs_learned 09_mnist_diagnostics --tier raw_priority --out research_experiments_v2/results/exports/v1_e8_e9_evidence.zip
```

Команда перечислит отсутствующие конкретные файлы и размеры. Старое дерево
только читается; ZIP и manifest с SHA256 пишутся исключительно в v2/results.
`audit-v1` также может пересчитать **описательные старые** парные дельты E3
из скаляров, с явной маркировкой unequal queries / one fixed distribution.
Это не восстанавливает исходные arrays и не создаёт новую независимую серию.
