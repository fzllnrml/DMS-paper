# MNIST dimension study: fixed protocol v1

Распакуйте `mnist-dimension-study.zip` в корень текущего репозитория поверх
предыдущих MNIST infrastructure/experiments/publication updates. Изменения
изолированы в `src/mnist_dimension/`. Старые Python-файлы, конфиги и результаты
не перезаписываются. Дополнительные зависимости не нужны.

```bash
source .venv-mnist/bin/activate
python -m pip install -r requirements-mnist-experiments.txt
python -m unittest discover -s tests -p 'test_mnist*.py' -v
```

## Входы и связь с полученным pixel result

Требуются только данные и обученные модели:

- `data/mnist/` с официальным MNIST cache
- `runs/mnist/checkpoints/best.pt` и завершённый `latest.pt` seed42
- `runs/mnist_seed43/checkpoints/best.pt`
- `runs/mnist_seed44/checkpoints/best.pt`
- `runs/mnist_evaluator/best.pt`

Старые search-output directories и aggregate CSV не требуются.
Выбран вариант B из задания: pixel-784 повторяется с прежними fixed definitions.
Предоставленные пользователем aggregate numbers не превращаются в искусственные
run-level rows и не используются для обучения, tuning или генерации CI.
Publication definition hash: `0de12232d9a259c1`, совпадает с предоставленными CSV.

## Scientific design

Основная кривая: unwhitened PCA-MNIST с d=16,32,64,128,256,512,784.
Один общий базис fit только на train, без labels, whitening и масштабирования
отдельных dimensions. Representations вложены: при росте d добавляются следующие
главные компоненты того же базиса. Данные исходно нормализуются в [-1,1].

PCA-784 и pixel-CNN-784 являются **двумя разными controls**. PCA-784 использует
полный ортонормальный базис и ту же MLP family, что другие PCA dimensions.
Pixel-784 использует существующую CNN и прежние DDIM/clamp definitions.
На графиках pixel point отделён маркером и не включён в PCA-линию.
PCA dimension означает число сохраняемых координат, а не доказанную intrinsic
dimension MNIST. Сохраняются eigenvalue spectrum и оценка численного rank.

| Representation | Dimensions | Model seeds | Search seeds | Trainings | Search suites |
|---|---|---|---|---:|---:|
| PCA, полная кривая | 16,32,64,128,256,512,784 | 42 | 0–4 | 7 | 35 |
| PCA, независимые replicas | 16,128,784 | 43,44 | 0–4 | 6 | 30 |
| Pixel CNN anchor | 784 | 42,43,44 | 0–4 | 0 новых | 15 |
| Всего | | | | 13 | 80 |

Один suite включает четыре парных метода. Всего 320 method runs.
Иерархический дизайн экономит 8 обучений относительно 7×3. Полная dimension
curve оценивается на seed42. Replicas в нижней, средней и полной размерности
проверяют, устойчивы ли ключевые различия к обучению модели. Промежуточные точки
остаются conditional-on-one-model, это ограничение не скрывается в CI.

В aggregation эти две задачи разделены: `curve_seed42` использует seed42 во
всех точках; `replicated_anchors` использует три model seeds только в anchors.
Нельзя соединять средние по трём моделям в одних d со средним по одной в других
и называть полученное изменение чистым dimension trend.

## Алгоритм и геометрия

Методы: `ddim`, `original_v3f2`, `rms_v3f2`, `matched_ddim`. Merge выключен.
P=128, DDIM=50 steps, alpha=0.03, levels=[800,500,300,200,100,50,20,5,0],
5 шагов на уровень, 10 final steps на t=0 с alpha×0.1. Прежний tolerance=1e-6.
При t=0 шум минимальный ненулевой. L2 и RMS отличаются только нормировкой шага:

- L2: Δ=0.03 s/||s||₂
- RMS: Δ=0.03 sqrt(d) s/||s||₂

Pixel projection остаётся clamp. PCA coefficients не имеют box [-1,1], поэтому
для PCA projection=none и DDIM denoised clipping выключен. Это различие явно
сохранено в definitions и является confound при сравнении PCA с pixel anchor.
Внутри PCA-кривой projection и sampling policy одинаковы для всех d.
Декодированные изображения не обрезаются перед evaluation. PNG обрезаются
только для отображения. Overshoot и доля пикселей за границей сохраняются.

Pairing: same model, dimension, search seed, Gaussian prefix и DDIM initialization.
Noise строится в 784 координатах и затем усекaется для PCA, что также сохраняет
общие случайные первые координаты между d. Для matched generation первые P
samples имеют те же batch boundaries. Сохраняются hashes Gaussian/DDIM prefixes.

Каждый метод выполняет и оплачивает собственную initialization. Matched count
равен floor(actual RMS NFE/50), а не заранее заданному числу. Записываются target,
actual, sample count и difference. При полном бюджете: 13440 RMS NFE,
268 matched samples, 13400 matched NFE, разность −40. NFE считает объекты,
не PyTorch batches. Wall time измеряется отдельно. Равный NFE между разными d
не означает равный FLOP budget, поэтому также записываются architecture и capacity.

## Центральная mechanistic diagnostic

До clipping, при ненулевом score выше tiny-norm guard:

| Величина | L2 | RMS |
|---|---|---|
| RMS шага на координату representation | 0.03/sqrt(d) | 0.03 |
| RMS декодированного pixel step для ортонормальной PCA | 0.03/28 | 0.03 sqrt(d/784) |

Вторая строка важна: постоянный RMS шаг в PCA coordinates не означает постоянный
RMS шаг по всем 784 пикселям. Измеряются обе шкалы, предложенные и фактические
шаги, final-step scale, net displacement, cumulative path length и displacement
относительно train-coordinate RMS. В raw stages остаются score norms, размеры
каждого шага и boundary statistics. Early stopping и clipping не маскируются.

Алгебраический control проверяет в unit test, что RMS(alpha) эквивалентен
L2(alpha sqrt(d)) на том же поле. Это не дополнительный tuned variant и не
требует новых scientific runs. Само совпадение с формулой 1/sqrt(d) не доказывает
причину выигрыша по support: оно гарантировано определением нормировки.

Основное evidence должно состоять в **изменении paired quality differences с d**,
согласованном с измеренным масштабом движения и повторяющемся на independent
model anchors. Его следует сопоставлять с reconstruction quality, generative
quality, diffusion loss, clipping и capacity. Pipeline не пишет «hypothesis confirmed».

## Representation и модели

PCA fit: потоковое накопление covariance в float64 на CPU, eigh матрицы 784×784,
согласование знаков eigenvectors, сохранение полного базиса в float32 для MPS.
Это уменьшает workspace по сравнению с full SVD матрицы 55000×784. Test/validation
только преобразуются, в fit не участвуют. Degenerate eigenspaces могут зависеть
от LAPACK/version, поэтому используется сохранённый базис и его hash, без refit.

Реконструкция оценивается на validation: retained train variance, RMS error,
conditional 95% image-bootstrap CI, per-image reconstruction errors, overshoot.
Representation-quality criterion не использует labels. Отдельно оцениваются
reconstructed real validation images теми же pixel/feature support metrics,
что generation: это показывает, насколько сама потеря информации меняет оценку.

MLP family переиспользует существующий PCADiffusion: scalar normalized time,
три hidden layers по 256, SiLU, epsilon prediction. Вход/выход меняются с d.

| d | Parameters |
|---:|---:|
| 16 | 140304 |
| 32 | 148512 |
| 64 | 164928 |
| 128 | 197760 |
| 256 | 263424 |
| 512 | 394752 |
| 784 | 534288 |

Это фиксированный capacity principle, не одинаковое число параметров. Рост
input/output layers фиксируется как возможный confound. CNN anchor имеет другую
архитектуру. Хорошее качество MLP заранее не предполагается.

Все PCA trainings используют одну recipe из завершённого seed42 latest.pt:
epochs, batch size, learning rate, AdamW weight decay, grad clip. Diffusion schedule
фиксирован: T=1000, beta_start=0.0001, beta_end=0.02. Проверяется одинаковый split.
Объектив: epsilon MSE, усреднённый по координатам. Инициализация независима для
каждого seed. CPU generators для shuffle/noise сохраняются в checkpoints.
Validation noise/t фиксированы и общие для всех model seeds; epsilon prefixes
общие для dimensions. Выбор checkpoint: минимум full-validation diffusion loss.
DataLoader работает с готовыми latents, num_workers=0.

Это та же optimizer/epoch recipe, но не точное воспроизведение CNN training code:
изменены representation, architecture и fixed-validation random design. Всё это
описано в resolved_config. Внутри PCA family recipe не зависит от d.

## Model-quality и support diagnostics

Для каждого из 16 моделей создаётся отдельный quality job:

- полный доступный training/validation history и convergence plot
- epsilon MSE на 256 validation images при t=[0,20,100,300,500,800,999]
- PCA MSE по общим первым 16 компонентам без дополнительного NFE
- 128 DDIM samples с одним заранее фиксированным sanity seed1729
- pixel/representation/feature kNN, support percentiles и semantic diagnostics
- conditional image-bootstrap CIs для validation loss и sanity feature distance

Prefix16 MSE помогает заметить «хороший» средний loss за счёт лёгких low-variance
tail coordinates. Epsilon MSE по всем координатам разных d не является идеальным
показателем качества распределения. Его нужно читать вместе с генерацией.
Если в старом pixel checkpoint нет history/best validation loss, это помечается,
а общий fixed-noise quality diagnostic всё равно выполняется заново.

Evaluation references: полный train, первые 1024 сохранённых validation indices,
k=5. Feature extractor — прежний frozen evaluator, split которого проверяется.
Representation distance = Euclidean/sqrt(d); дополнительно делится на RMS масштаба
train coordinates. Support percentile калибруется через validation→train в той же
representation. Pixel/feature diagnostics после decode используют общий исходный
MNIST train reference. Сырые Euclidean distances разных d напрямую не ранжируются.

kNN исчерпывающий, без ANN: float64 block matrix multiplication на CPU,
near-copy pairs и итоговые выбранные расстояния пересчитываются прямым вычитанием.
Он проверен против прямого cdist, включая near-copy examples. Это может изменить
последние знаки округления относительно старого float32 evaluation, но не
математическое определение расстояния. Старый evaluator-код не меняется.
Reference features/distances кэшируются, classifier forward даёт logits и features
одновременно. Search не получает labels или classifier information.

## Failure visibility и uncertainty

Плохие, но finite результаты не исключаются. Флаги имеют заранее заданные
диагностические пороги: retained variance<0.9, confidence<0.5, outside/clipping>0.25,
ухудшение validation loss относительно первой эпохи, отсутствие history.
Они не используются для выбора параметров или исключения dimensions.
NaN/Inf, missing/corrupt artifacts и runtime errors видны в job inventory.

Каждый ожидаемый dimension point остаётся в scientific table, даже если все его
jobs отсутствуют/завершились ошибкой. Completed methods из partial suite остаются
в raw table. Primary summaries используют только целиком completed paired suites,
с указанием expected/completed counts. Это предотвращает незаметное превращение
неполной пары в независимые наблюдения. Отсутствующие точки отмечены на графиках.

Статистика: mean, std, median, n, 95% bootstrap CI, paired RMS−L2 и RMS−matched.
В anchors сначала ресэмплируются model seeds, затем search seeds внутри модели.
На полной seed42 curve uncertainty условна на один checkpoint. Quality tables
имеют одну строку на модель и не размножают validation loss по search seeds.
Representation/model image-bootstrap CIs условны на фиксированный basis/model.
PCA fitting variability, architecture confounding и intrinsic dimension ими
не покрываются. P-values и автоматические научные выводы не формируются.

## Compute plan

| Часть | Количество | Inference NFE upper bound |
|---|---:|---:|
| Общий representation fit | 1 | 0 |
| PCA diffusion trainings | 13 | training budget учитывается отдельно |
| PCA paired suites | 65 | 3034200 |
| Pixel paired suites | 15 | 700200 |
| Все search suites | 80 | 3734400 |
| Model-quality jobs | 16 | 131072 |
| Итого inference | | 3865472 |

На один search suite: 6400 DDIM + 13440 L2 + 13440 RMS + 13400 matched = 46680.
На одну model-quality job: 256×7 validation evaluations + 128×50 DDIM = 8192.
В search нет дополнительных stability/curvature forwards. Quality NFE хранится
отдельно и не умножается повторно на пять search seeds.

Training epochs неизвестны до чтения локального recipe checkpoint. Например,
при 30 эпохах и split55000/5000 получится 23.4 млн train+validation forward-object
passes для 13 моделей, плюс backward на training portion. Это не включено в
inference NFE. CLI dry-run не читает checkpoint и не выдаёт training cost за ноль.

Оценка диска: порядка 1–2 GB новых artifacts, резерв 2 GB сверх существующего
MNIST cache/checkpoints. Latents всех splits при max d784 занимают около 220 MB,
raw search populations около 235 MB, step tensors около 23 MB; остальное —
checkpoints с optimizer/basis, immutable copies, cache, JSON и figures.
Это оценка, не жёсткий лимит. Full trajectories не сохраняются, grids ограничены
16 изображениями и search seed0. Все численные step diagnostics сохранены.

Ожидаемый основной bottleneck — 13 обучений. В evaluation заметна стоимость
исчерпывающего CPU kNN относительно 55000 train images. Covariance и kNN float64
остаются на CPU, нейросети используют MPS/CUDA/CPU. Оценка часов без запуска
на пользовательском MacBook не даётся.

## Команды

Все команды из корня репозитория с активированным `.venv-mnist`.

Предварительный smoke на малом подмножестве локального MNIST, отдельный root
`runs/mnist_dimension_smoke`, 2/4 PCA coordinates, два tiny MLP, P=2, один seed,
одна эпоха. Он использует существующие pixel/evaluator checkpoints:

```bash
python -m src.mnist_dimension.runner smoke --execute --device mps
```

1. Подготовить representation и обучить 13 PCA-моделей:

```bash
python -m src.mnist_dimension.runner train --execute --device mps
```

2. Resume обучения:

```bash
python -m src.mnist_dimension.runner train --execute --resume --device mps
```

3. Посмотреть полный compute plan без чтения данных/checkpoint:

```bash
python -m src.mnist_dimension.runner search --dry-run
```

Для training plan также работает `train --dry-run`. Без `--execute` train/search
вообще не выполняют вычислительные jobs.

4. Model-quality checks и все 80 search suites:

```bash
python -m src.mnist_dimension.runner search --execute --device mps
```

5. Resume search:

```bash
python -m src.mnist_dimension.runner search --execute --resume --device mps
```

6. Таблицы, Markdown и графики одной командой:

```bash
python -m src.mnist_dimension.runner aggregate
```

7. Отдельная команда графиков не нужна: пункт6 создаёт PNG/PDF автоматически.
По умолчанию bootstrap=2000, seed=123. `--bootstrap 2000` задаёт repeats явно.

## Файлы и безопасный resume

Root: `runs/mnist_dimension/`.

- `study_plan.json`: ожидаемые jobs и resolved study configuration
- `representation/`: basis.pt, общие latents.pt, reconstruction quality
- `models/pca_d*_m*/`: best.pt, latest.pt, history.json, training config
- `quality/`: independent generation/validation diagnostics каждой модели
- `search/`: suite/method summaries, raw arrays, step measurements и grids
- `checkpoints/`: неизменяемые SHA-копии использованных моделей
- `evaluation_cache/`: train features и reference distances
- `summary/raw_runs.csv`, `dimension_summary.csv`, `paired_comparisons.csv`
- `summary/model_representation_quality.csv`, `scientific_table.csv`, `job_status.csv`
- `summary/dimension_summary.md`, JSON counterparts и `plots/*.png,*.pdf`

Каждая стадия имеет started/completion metadata, code/config/dependency hashes
и checksums готовых artifacts. Сохраняются git commit/dirty state, Python/package
versions, модель, seed, basis/data hash и фактические численные параметры.
Completed jobs пропускаются до inference; изменение входов или кода блокирует
небезопасный resume. Не редактируйте код/конфиг во время серии. Используйте один
runner на root и тот же device argument. Старые outputs не используются как root.

Training resume воспроизводит последнюю полную эпоху, включая shuffle/noise RNG
и optimizer. Незавершённая эпоха выполняется заново. Search resume сохраняет
готовые методы, незавершённый метод повторяет с теми же seeds. После ошибок runner
сохраняет invocation inventory, пытается обработать остальные независимые jobs
и завершает процесс с ненулевым кодом при наличии failed/blocked jobs.

## Аудит прежнего кода и bugs

Переиспользованы PCADiffusion/MLP, DDIM, batched NFE wrapper, movement implementation,
publication method definitions, frozen evaluator, serialization и bootstrap.
Старый PCA-код был train-only и unwhitened, но сам по себе не обеспечивал frozen
replication design, model-quality diagnostics и failure visibility, mechanistic tables и корректное
разделение seed42 curve и independently trained anchors. Новый pipeline добавляет
это отдельно. PCA fit переведён на covariance/eigh для меньшего workspace.

Найдено окно сбоя в старом общем PCA/evaluator trainer: он сначала сохраняет
latest.pt, затем best.pt. При interruption строго между этими записями latest
может содержать обновлённый best_loss при устаревшем best.pt. Новый dimension
trainer сохраняет best раньше latest и проверен на exact resume.

Старый helper намеренно не перезаписан: иначе изменился бы code hash уже
выполненной publication series. Это не ошибка RMS/L2/search/NFE. Обычный
непрерывавшийся запуск ей не затронут. По предоставленным агрегатам невозможно
установить, было ли такое interruption у старого evaluator/PCA training.
При подозрении на него нужно сверить best/latest и историю выбора checkpoint,
а не автоматически пересчитывать все полученные MNIST results.

Все файлы двух предыдущих обновлений побайтово сохранены. Проверен неизменный
publication code_digest, поэтому установка этого пакета не меняет его resume
semantics. Synthetic/Levine source также не изменён.

## Границы выполненной проверки

Не запускались научные trainings, полная dimension series, дополнительные
hyperparameter sweeps или поиск желаемого тренда. Не анализировались заново
отсутствующие local run-level data. Числа из user context приняты как завершённые
результаты. Pipeline только подготавливает следующий научный эксперимент.

Выполнены unit tests и tiny smoke с искусственными MNIST-shaped данными.
CLI smoke использует обычный torchvision loader и маленькие локальные IDX files,
малые MLP и случайную CNN fixture. Это проверка кода, не научный результат.
MPS/CUDA в текущей среде отсутствуют, аппаратные tests пропущены.

## Проверки и их результаты

36 passed, 2 hardware skips. Ниже полный результат каждого теста:

```text
test_actual_training_resume_exact (test_mnist_dimension.DimensionTests.test_actual_training_resume_exact) ... pca_d2_m42 epoch 2/2: train=0.744099 val=1.57443
pca_d2_m42 epoch 1/2: train=1.41406 val=1.57544
pca_d2_m42 epoch 2/2: train=0.744099 val=1.57443
ok
test_end_to_end_smoke_quality_failure_visibility_and_resume (test_mnist_dimension.DimensionTests.test_end_to_end_smoke_quality_failure_visibility_and_resume) ... representation fit
pca_d2_m42
pca_d2_m42 epoch 1/1: train=1.41406 val=1.57544
pca_d4_m42
pca_d4_m42 epoch 1/1: train=1.22963 val=1.6425
quality pca_d2_m42
pca_d2_m42_s0
quality pca_d4_m42
pca_d4_m42_s0
quality pixel_d784_m42
pixel_d784_m42_s0
quality pca_d2_m42
pca_d2_m42_s0
quality pca_d4_m42
pca_d4_m42_s0
quality pixel_d784_m42
pixel_d784_m42_s0
quality pca_d2_m42
pca_d2_m42_s0
FAILED pca_d2_m42_s0: nonfinite score fixture
quality pca_d4_m42
pca_d4_m42_s0
quality pixel_d784_m42
pixel_d784_m42_s0
quality pca_d2_m42
pca_d2_m42_s0
quality pca_d4_m42
pca_d4_m42_s0
quality pixel_d784_m42
pixel_d784_m42_s0
ok
test_fixed_design_and_budget (test_mnist_dimension.DimensionTests.test_fixed_design_and_budget) ... ok
test_heldout_images_and_labels_cannot_change_fit (test_mnist_dimension.DimensionTests.test_heldout_images_and_labels_cannot_change_fit) ... ok
test_knn_exhaustive_near_copy (test_mnist_dimension.DimensionTests.test_knn_exhaustive_near_copy) ... ok
test_measured_movement_and_decoded_scale (test_mnist_dimension.DimensionTests.test_measured_movement_and_decoded_scale) ... ok
test_paired_bootstrap_and_strata (test_mnist_dimension.DimensionTests.test_paired_bootstrap_and_strata) ... ok
test_pixel_definition_and_scaling_control (test_mnist_dimension.DimensionTests.test_pixel_definition_and_scaling_control) ... ok
test_train_only_fit_and_inverse (test_mnist_dimension.DimensionTests.test_train_only_fit_and_inverse) ... ok
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
Ran 38 tests in 29.685s

OK (skipped=2)
```

CLI train/search dry-run прошли. CLI smoke execute и следующий resume
на финальной версии кода прошли с exit code 0. PNG/PDF проверены на
малых искусственных данных. Archive содержит только код, config, tests и runbook,
без искусственных результатов или обученных fixtures.
