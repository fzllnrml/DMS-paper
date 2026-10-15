# Fixed RMS-v3f2 publication protocol

Распакуйте `mnist-publication.zip` в корень своей текущей копии репозитория,
поверх предыдущего `mnist-experiments.zip`. Обученные checkpoint и старые результаты
в архив не входят. Требуется существующий `src/mnist/train.py` из первого MNIST-этапа.
Новые зависимости не добавлены.

```bash
source .venv-mnist/bin/activate
python -m pip install -r requirements-mnist-experiments.txt
python -m unittest discover -s tests -p 'test_mnist*.py' -v
```

## Что зафиксировано

Протокол `mnist_rms_publication_v1` хранится в `publication.py`. Runner запрещает
менять математическое определение методов через publication YAML. Сохраняются
его hash и полные определения. Старые exploratory-конфиги остаются отдельными.

Общее начало: unconditional pixel CNN, `[-1,1]`, DDIM eta=0, 50 steps.
Движение: `[800,500,300,200,100,50,20,5,0]`, 5 шагов на уровень,
10 final steps на t=0, alpha=0.03, final scale=0.1, clamp, прежний tolerance=1e-6.
При t=0 шум минимальный ненулевой. Это не точный score чистого распределения.

| Метод | Определение |
|---|---|
| `ddim` | P DDIM samples, без движения/merge |
| `ddim_cluster` | DDIM-P, только final medoid merge |
| `original_v3f2` | L2-нормировка, тот же alpha/бюджет, без merge |
| `rms_v3f2` | RMS-нормировка, без merge |
| `rms_v3f2_merge` | RMS + final-only medoid merge |
| `matched_ddim` | DDIM по фактическому NFE `rms_v3f2` |
| `matched_cluster` | DDIM + merge по фактическому NFE `rms_v3f2_merge` |

Merge: complete linkage, RMS Euclidean, constant radius=0.65, medoid.
L2-контроль здесь воспроизводит thesis-style формулу шага при одинаковом бюджете
с RMS. Он не заявлен как полное воспроизведение всех thesis hyperparameters.
Полный прежний `original` с его настройками сохранён в exploratory runner.

Компактная ablation: DDIM, L2, RMS, RMS+medoid, RMS+mean и RMS+every-level medoid.
Каждый из последних двух отличается от основного merge-варианта одним фактором.
Альфа и радиус нигде не перебираются. PCA в новые серии не входит.

## Парность и NFE

Каждый метод оплачивает собственную DDIM initialization, которая действительно
выполняется. Seed и Gaussian prefix совпадают. Для matched methods границы
батчей в первых P samples сохраняются, поэтому DDIM prefix также совпадает.
Сохраняются оба hash, model/search/noise seeds, checkpoint hash и paired_suite_id.
Это исключает расхождения первых P samples из-за другой формы батчей.

Сначала выполняются target methods, затем matched baselines:
`N = floor(actual_target_NFE / 50)`. Записываются `target_v3f2_nfe`,
`matched_target_method`, `matched_ddim_sample_count`, `matched_ddim_nfe`,
`nfe_difference = matched_NFE - target_NFE`. Разность принадлежит (-50, 0].
Объединение после всех score-шагов не добавляет diffusion NFE и не сокращает
число уже выполненных движений. В стандартном случае P128 даёт 13440 против
13400 NFE и 268 matched samples. При early stopping берётся фактический бюджет.

NFE считает объекты, не batch calls. `wall_time` включает DDIM+movement+merge,
но исключает evaluation. Diagnostic NFE, evaluation time и classifier work
учитываются отдельно. Честный compute total: search NFE + diagnostic NFE.

## Evaluation

Во всех новых сериях: полный train reference, первые 1024 validation indices
из сохранённого split, k=5. Evaluator фиксирован: `runs/mnist_evaluator/best.pt`.
Train/validation split evaluator должен совпадать со split diffusion model.
Pixel distances выражаются в RMS нормализованных пикселей, feature distances в
L2 единицах frozen evaluator. Reference features и validation distances
кэшируются по данным, evaluator hash, версии PyTorch и устройству evaluator.
Повторная оценка идентичных image tensors внутри suite использует кэш.

Memorization audit включён: nearest train/test distances и их разность для
каждого кандидата. Порог «необычно близко к train» равен 1%-квантилю
validation→train nearest distances. Сохраняются доля таких кандидатов, reference
distributions и percentile разности train−test относительно validation.
Это учитывает разный размер train/test references при интерпретации. Диагностика
сама по себе не доказывает ни копирование, ни его отсутствие. Test не влияет
на движение, merge, выбор checkpoint или настройку параметров.

Main stability: DDIM, RMS, RMS+merge, matched-DDIM; все final candidates,
2 направления, RMS perturbations 0.005 и 0.01, 10 local steps, RMS alpha=0.003,
t=0, return radius=0.01. Сохраняется также дрейф непотревожденного кандидата.
Diagnostic upper bound равен 50 score evaluations на кандидата.
В ablation/population stability выключена. Curvature остаётся доступна в старом
коде, по умолчанию в новых сериях выключена.

Терминология: mode candidates / high-support candidates. Классы цифр используются
только как semantic diagnostics и не считаются математическими модами.

## Матрицы и стоимость

Один job представляет парный suite, а не один метод.

| Серия | Jobs | Запусков методов | Search NFE upper bound | Diagnostic NFE upper bound | Всего NFE upper bound |
|---|---:|---:|---:|---:|---:|
| `publication_train` | 2 | 2 обучения | не применимо | не применимо | зависит от source recipe |
| `publication_main` | 30 | 210 | 1 798 200 | 733 500 | 2 531 700 |
| `publication_ablation` | 3 | 18 | 220 800 | 0 | 220 800 |
| `publication_population` | 12 | 48 | 604 200 | 0 | 604 200 |

Main: models42/43/44 × search seeds0–4 × P64/128. Главная точка P128.
Ablation: model42 × seeds0–2, P128. Population: model42 × seeds0–2 × P32/64/128/256.
Самая дорогая поисковая серия main. Два обучения могут занять значительно
больше времени, чем поиск. Их forward/backward стоимость не выдаётся за ноль:
поле search NFE в training dry-run к обучению не относится.

Оценка raw image/endpoint tensors: main 599 MB, ablation 36 MB, population
140 MB (десятичные MB). Это около 775 MB суммарно без JSON, служебных tensors,
PNG/PDF и копий checkpoint. Практический резерв для новых поисковых результатов:
примерно 2 GB, плюс два обученных запуска и существующий MNIST cache.
Это оценка, не жёсткий лимит. Полные trajectories выключены. Grids сохраняются
только для search seed0, по 16 строк. Figure code headless, PNG и PDF.
Общая reference feature cache занимает около 14 MB при features64 и train55000.

## Готовые команды

Все команды из корня репозитория с активированным `.venv-mnist`.
Без `--execute` runner всегда делает dry-run. Ни один dry-run не читает MNIST,
checkpoint или evaluator и не создаёт scientific outputs.

A. Обучить только seeds43 и44:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_train.yaml --execute --device mps
```

Конфиг берётся из завершённого `runs/mnist/checkpoints/latest.pt` seed42.
Это сохраняет реальный общий бюджет эпох даже если `best.pt` появился до
дообучения. Архитектура и split сверяются с `best.pt`. Требуются оба файла.
Меняются только seed, output directory и device. Используется тот же существующий
trainer, normalization, optimizer, fixed validation procedure и выбор best loss.
Новые модели инициализируются независимо, веса seed42 не передаются trainer.
После обучения проверяются конфиг, architecture, split и завершение бюджета.
Выходы: `runs/mnist_seed43/` и `runs/mnist_seed44/`, с `checkpoints/best.pt`,
`checkpoints/latest.pt` и прежними training logs.

B. Продолжить после interruption:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_train.yaml --execute --resume --device mps
```

Продолжается с `latest.pt` каждой модели. Completed models пропускаются.
После сбоя до первого checkpoint неполный training directory сохраняется
с суффиксом `.interrupted-<timestamp>`, затем этот seed начинает обучение заново.

C. Посмотреть main до вычислений:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_main.yaml --dry-run
```

D. Запустить main:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_main.yaml --execute --device mps
```

E. Продолжить main:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_main.yaml --execute --resume --device mps
```

F. Компактная ablation после проверки main:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_ablation.yaml --execute --device mps
```

G. Population/NFE curves:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication_population.yaml --execute --device mps
```

К F/G также можно добавить `--resume`; `--dry-run` работает для всех четырёх YAML.

H. Aggregation основного результата:

```bash
python -m src.mnist_experiments.publication_aggregate --root runs/mnist_experiments/publication_main --output runs/mnist_experiments/publication_summary
```

После выполнения дополнительных серий можно собрать их отдельные таблицы и
графики той же командой, без смешивания результатов между сериями:

```bash
python -m src.mnist_experiments.publication_aggregate --root runs/mnist_experiments/publication_main runs/mnist_experiments/publication_ablation runs/mnist_experiments/publication_population --output runs/mnist_experiments/publication_summary
```

I. Final plots создаются командой H автоматически. Отдельная команда не нужна.
`--no-plots` выключает только построение графиков, `--bootstrap 2000 --seed 123`
явно задают значения по умолчанию.

## Artifacts, statistics и resume

Каталоги новых серий: `runs/mnist_experiments/publication_<series>/`.
Каждый suite сохраняет resolved YAML, metadata, immutable checkpoint copies,
method arrays/metrics/summary, completion markers. Summary содержит unified metric
names, actual NFE, budgets, timing, clipping, model/search seeds и hashes.
Triplets и paired grids имеют фиксированный ancestor indexing. Merged representative
может повторяться в нескольких строках, mapping/названия колонок записаны в JSON.

Aggregation читает только целиком завершённые suites, проверяет method markers,
не подмешивает partial jobs и не заполняет пропуски. Выходы:
`raw_runs.csv`, `summary.csv`, `paired_comparisons.csv`, `publication_summary.md`,
их JSON версии, bootstrap config и `plots/`. Markdown создаётся только после
вашего локального запуска. Это таблицы чисел, без автоматически написанных выводов.

Статистика: n_runs, mean, std, median, 95% hierarchical bootstrap CI. Сначала
ресэмплируются model seeds, затем search seeds внутри каждой модели. В paired
comparisons ресэмплируются парные разности. Знак всегда «method minus reference».
Основные contrasts: RMS−DDIM, RMS−matched, RMS+merge−DDIMcluster,
RMS+merge−matchedcluster. Дополнительные contrasts выделяют L2/merge/mean/schedule.
Серии, populations, evaluator и definitions группируются отдельно. Дубликаты и
нарушение парности отклоняются. Один model seed означает conditional uncertainty,
три model seeds дают ограниченную точность оценки model-level variability.
P-values не вычисляются.

Search resume требует те же config/code/checkpoint hashes. Completed methods
пропускаются до повторной inference. Незавершённый метод запускается заново
с теми же seeds, это не resume с произвольного score step. Код не перезаписывает
готовые результаты при отказе в resume. Запускайте один runner на каталог серии.
Resume старых exploratory jobs требует соответствующей старой версии кода,
поскольку их code hash изменится при установке этого обновления. Уже сохранённые
численные результаты остаются пригодны для чтения и агрегации.

## Исправления и границы проверки

1. Старый `aggregate.py` пропускал `feature_knn_mean` в summary/paired tables,
   хотя raw metrics содержали значение. Оно добавлено. Это не затрагивало search,
   кандидаты или raw feature distances. Для старых таблиц достаточно заново
   выполнить старую aggregation command. Повторный поиск не требуется.
2. Runner раньше мог записать `failure.json` в уже существующий каталог даже
   при отказе из-за несовместимого resume или существующих outputs. Такие отказы
   больше не меняют каталог. Прежние numerical results это не затрагивало.

Synthetic/Levine source и математические определения старого MNIST search,
DDIM и PCA не изменены. Publication-варианты добавлены отдельным протоколом.

Не запускались training43/44, publication experiments, sweeps, PCA training,
большие matched-NFE расчёты. Локальные обученные checkpoint пользователя здесь
недоступны. Проверка обучения seeds касается нового orchestration: точного
копирования recipe, передачи resume, проверки artifacts и skip completed.
В тесте subprocess существующего trainer заменён контролируемой fixture.
Полное обучение этим тестом не имитируется и не заявляется проверенным здесь.

CPU smoke использует маленькую случайно инициализированную CNN, два исходных
изображения, все методы и искусственный dataset. Сохраняется полный фиксированный
time/step protocol. MPS/CUDA тесты пропускаются при отсутствии оборудования.
Побитовое совпадение между backend/версиями библиотек не гарантируется.

## Результаты проверки кода

27 tests passed, 2 hardware skips. Full log:

```text
test_all_distances (test_mnist_experiments.ExperimentsTests.test_all_distances) ... ok
test_complete_linkage_mean_medoid (test_mnist_experiments.ExperimentsTests.test_complete_linkage_mean_medoid) ... ok
test_cuda (test_mnist_experiments.ExperimentsTests.test_cuda) ... skipped 'CUDA unavailable'
test_ddim_pairing_and_checkpoint (test_mnist_experiments.ExperimentsTests.test_ddim_pairing_and_checkpoint) ... ok
test_full_tiny_suite_resume (test_mnist_experiments.ExperimentsTests.test_full_tiny_suite_resume) ... ok
test_image_counted_batched_score (test_mnist_experiments.ExperimentsTests.test_image_counted_batched_score) ... ok
test_legacy_synthetic_levine (test_mnist_experiments.ExperimentsTests.test_legacy_synthetic_levine) ... ok
test_matrix_dry_run (test_mnist_experiments.ExperimentsTests.test_matrix_dry_run) ... ok
test_merge_schedules_projection_radius (test_mnist_experiments.ExperimentsTests.test_merge_schedules_projection_radius) ... ok
test_mps (test_mnist_experiments.ExperimentsTests.test_mps) ... skipped 'MPS unavailable'
test_original_matches_legacy (test_mnist_experiments.ExperimentsTests.test_original_matches_legacy) ... ok
test_paired_statistics (test_mnist_experiments.ExperimentsTests.test_paired_statistics) ... ok
test_pca_transform_inverse (test_mnist_experiments.ExperimentsTests.test_pca_transform_inverse) ... ok
test_semantics_knn_percentiles (test_mnist_experiments.ExperimentsTests.test_semantics_knn_percentiles) ... ok
test_stability_curvature (test_mnist_experiments.ExperimentsTests.test_stability_curvature) ... ok
test_step_scaling (test_mnist_experiments.ExperimentsTests.test_step_scaling) ... ok
test_time_schedules (test_mnist_experiments.ExperimentsTests.test_time_schedules) ... ok
test_tiny_evaluator_train_resume (test_mnist_experiments.ExperimentsTests.test_tiny_evaluator_train_resume) ... ok
test_tiny_pca_train_resume (test_mnist_experiments.ExperimentsTests.test_tiny_pca_train_resume) ... ok
test_bootstrap_paired_validation (test_mnist_publication.PublicationTests.test_bootstrap_paired_validation) ... ok
test_fixed_definition_and_reject_tuning (test_mnist_publication.PublicationTests.test_fixed_definition_and_reject_tuning) ... ok
test_full_publication_smoke_resume_cache_aggregation (test_mnist_publication.PublicationTests.test_full_publication_smoke_resume_cache_aggregation) ... [1/1] bad
ok
test_legacy_feature_distance_aggregation (test_mnist_publication.PublicationTests.test_legacy_feature_distance_aggregation) ... ok
test_matched_calculation (test_mnist_publication.PublicationTests.test_matched_calculation) ... ok
test_matrix_sizes_paths_costs (test_mnist_publication.PublicationTests.test_matrix_sizes_paths_costs) ... ok
test_memorization_reference_calibration (test_mnist_publication.PublicationTests.test_memorization_reference_calibration) ... ok
test_seed42_completed_training_recipe (test_mnist_publication.PublicationTests.test_seed42_completed_training_recipe) ... ok
test_seed_training_recipe (test_mnist_publication.PublicationTests.test_seed_training_recipe) ... ok
test_training_driver_interrupt_resume_skip (test_mnist_publication.PublicationTests.test_training_driver_interrupt_resume_skip) ... ok

----------------------------------------------------------------------
Ran 29 tests in 16.721s

OK (skipped=2)
```

После последней совместимой с Matplotlib 3.8 правки повторно прошёл полный
publication smoke (включая PNG/PDF и aggregation). CLI execute/resume на P=2
прошёл. CLI dry-run каждой из четырёх матриц прошёл. `git diff --check` прошёл,
исходные synthetic/Levine файлы побайтово совпадают с исходным checkout.
