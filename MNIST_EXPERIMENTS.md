# MNIST: экспериментальный код поиска mode candidates

Этот пакет дополняет предыдущий `mnist-infrastructure.zip`. Распакуйте его в корень
своей текущей копии `diffusion-mode-search`, сохранив папку `runs/`. Заменяется
`src/mnist/model.py`: архитектура и имена весов прежние, у `sample_ddim` добавлен
необязательный `initial_noise`. Обученный `runs/mnist/checkpoints/best.pt`
переобучать для запуска поиска не требуется. Synthetic/Levine source не изменён.

## Установка и первая команда

Все команды ниже выполняются из корня репозитория. Используйте существующее
окружение MNIST с ARM64 Python на Apple Silicon:

```bash
source .venv-mnist/bin/activate
python -m pip install -r requirements-mnist-experiments.txt
python -m unittest discover -s tests -p 'test_mnist_experiments.py' -v
```

`pytest` не требуется. Для CPU/CUDA замените `--device mps`. MPS/CUDA tests
автоматически пропускаются при отсутствии оборудования. Полноценные модели,
scientific sweeps и обученный пользовательский checkpoint в этой работе не запускались.
Тесты используют аналитические поля и очень маленькие искусственные наборы.

**COMMAND 1:** дешёвая проверка текущего обученного checkpoint на MPS:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/diagnostic.yaml --execute --device mps
```

Это один job, 8 initial images, 20 DDIM steps, уровни `[50,15,0]`, по 2 шага
движения и 2 final steps. Внутри четыре метода: original, DDIM, DDIM+cluster,
DDIM+final refinement. Для support используются 1024 train и 64 validation images.
Evaluator не требуется. Эти числа достаточны только для диагностики pipeline.
Данные должны быть уже скачаны предыдущим обучением (`data/mnist`).

Просмотр всей будущей матрицы **без вычислений и без чтения датасета**:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/all.yaml --dry-run --plan-output runs/mnist_plan.json
```

В поставленных presets 155 jobs, из них 10 обучений: 1 evaluator, 7 PCA/MLP,
2 дополнительных pixel CNN. Также есть 1 PCA fit. Текущая pixel-модель seed 42
не переобучается. Это список возможностей для последовательной работы,
а не рекомендация сразу выполнить всю матрицу. Без `--execute` runner всегда
делает только dry-run. Стоимости поиска показаны как верхние оценки без merge/
early stopping. Evaluator forward и PCA fit в diffusion NFE не входят.

## Архитектура

| Файл в `src/mnist_experiments/` | Назначение |
|---|---|
| `common.py`, `config.py` | Контракты, seed, NFE, serialization, defaults |
| `geometry.py`, `search.py` | Movement, complete-linkage merge, trajectories, baselines |
| `data.py`, `evaluation.py` | Исходные split indices, kNN/support, semantics, stability, curvature |
| `evaluator.py`, `training.py` | Независимый classifier и обучение с epoch resume |
| `pca.py` | Train-only PCA, transform/inverse, MLP diffusion |
| `runner.py` | YAML matrix expansion, opt-in execution, resume |
| `aggregate.py`, `plots.py` | Статистика, paired CI, PNG plots |

Search state: float32 `[P,d]`. Pixel model получает батч через
`MNISTDiffusion.score(x_batch,t_batch)`, DDIM через `sample_ddim`.
Ни один метод не вызывает сеть отдельно для каждой частицы в цикле. Большие
батчи делятся по `batch_size`; последний батч может содержать одну частицу.
NFE считается по изображениям во всех таких вызовах. Фиксируются seeds и CPU
RNG для initialization/training. Между CPU, MPS, CUDA и версиями PyTorch
побитовое совпадение не гарантируется.

## Original и контролируемые изменения

`original` сохраняет порядок исходного v3f2:
DDIM population → движение на фиксированном t → complete-linkage merge →
следующий t → final refinement → final merge. Complete linkage, а не single
linkage/connected components. Mean усредняет текущие представители с равными
весами, как старый код. Число исходных частиц хранится отдельно в lineage.
На аналитическом score проверяется совпадение результатов и NFE со старым v3f2.

Default: L2-normalized score, alpha=0.01, 20 шагов/уровень, final=100 с alpha×0.1,
уровни `[800,500,300,200,100,50,20,5,0]`, Euclidean, every-level mean merge,
clamp. Радиус `max(0.05, 0.8*sigma_t)`; для t=0 сохранена историческая подстановка
0.01. DDIM использует существующий MNIST sampler с корректным переходом к x0,
не старый векторный helper. Diagnostic сокращает бюджеты и число уровней,
но сохраняет эти механизмы.

`method=original` означает основной цикл v3f2. Если geometry config изменён,
это отдельный `variant/original`, а не результат faithful default. Имена variant,
полный config и condition_id сохраняются. Pixel и feature-варианты не объединяются
при aggregation.

| Параметр | Варианты и смысл |
|---|---|
| `step_mode` | `l2`: αs/‖s‖; `rms`: α√d s/‖s‖; `raw`: αs |
| `schedule.kind` | `explicit` + `timesteps`; `sigma` / `log_snr` + `values` или `min,max,count` |
| `merge.metric` | `euclidean`, `rms_euclidean`, `cosine`, `feature_l2`, `feature_cosine` |
| `merge.schedule` | `every_level`, `final_only`, `selected_levels` + список t, `none` |
| `merge.radius_schedule` | `constant` + `radius`; `noise_scaled` + `coefficient,min_radius,noise_units` |
| `merge.representative` | `mean`, `medoid` (минимум суммы внутрикластерных расстояний) |
| `projection` | `clamp`, `none`; для PCA только `none` |
| `initialization` | `ddim`; отдельно `uniform` в pixel box как ablation |

Для sigma levels используется геометрическая сетка, для log-SNR линейная.
Ближайшие дискретные t и число схлопнувшихся уровней записываются в metadata.
**t=0 означает минимальный ненулевой шум. Это не точный score чистого p0.**
Направление шага использует тот же tiny-norm guard, что исходный v3f2.
Логируются предложенные и фактические L2/RMS шаги, score norms, выход за box
до clipping и displacement относительно средних исходных предков.

Дополнительные обоснованные проверки:

- `noise_units=pairwise_sigma`: радиус масштаба σ√(2d) в L2 или σ√2 в RMS,
  естественного для разности двух независимых Gaussian noises. Default остаётся
  `legacy_sigma`. Для cosine/feature distances такой физический масштаб не
  определён; используйте constant radius и отдельную калибровку.
- Stability также уточняет непотревоженный центр и показывает его drift.
  Это помогает отличить невозврат perturbations от движения самого кандидата.
- Distance statistics включают CV и quantiles. Без merge диагностика использует
  не более 512 равномерно выбранных частиц, с явным `approximate=true`.
  Сам clustering никогда не заменяется приблизительным.

Cosine на исходных пикселях [-1,1] чувствителен к общему фону. Feature geometry
использует дополнительную обученную сеть. Ни один из вариантов не считается
улучшением до запланированных сравнений.

## Baselines и бюджет

| Имя | Действие |
|---|---|
| `ddim` | P ordinary DDIM samples |
| `ddim_cluster` | Те же samples + один final merge, без движения |
| `final_refine` | DDIM + final low-noise refinement, без merge |
| `no_intermediate_merge` | Все уровни движения, merge только в конце |
| `low_noise_only` | Весь многоуровневый бюджет движения перенесён на final_t |
| `matched_ddim` | floor(NFE_original / ddim_steps) ordinary samples |
| `matched_cluster` | Matched DDIM samples + тот же final merge |

Original выполняется первым. Matched baselines не превышают его actual search
NFE; остаток меньше одного DDIM sample и сохраняется. Их starts используют общий
префикс explicit Gaussian noise. Seed по умолчанию равен search_seed, а не одному
фиксированному seed для всех повторов. Бюджет каждой пары учитывает собственную
DDIM initialization, которую код действительно выполняет.

`nfe` = initialization + movement. `diagnostic_nfe` = stability + curvature.
Также сохраняются total diffusion NFE, число batched forward calls и evaluator
images, включая feature geometry. Wall time поиска включает его instrumentation.
CUDA peak allocation измеряется отдельно; MPS поле является snapshot, не peak.
RSS обозначен как peak процесса, не изолированного метода.
Complete-linkage имеет квадратичную стоимость: `merge.max_particles=4096` защищает
от случайного большого выделения памяти. `max_matched_samples=10000` ограничивает
генерацию. При превышении код выдаёт ошибку, а не урезает эксперимент молча.

## Evaluation и интерпретация

Все результаты называются **mode candidates**. Десять digit labels не являются
десятью истинными модами. Frozen evaluator выдаёт class/confidence, histogram,
coverage и entropy. Его validation accuracy записывается в metadata. В pixel
experiments он используется только для внешней оценки. В `feature_*` участвует
в геометрии поиска, что отдельно отмечается в результатах.

kNN exhaustive и обрабатывается блоками CPU. Опорные точки берутся только из train
исходного diffusion split. Pixel support использует RMS distance, feature support
обычный L2. Сохраняются nearest distance, среднее по k соседям, индексы, расстояния
и reference distribution validation→train. `distance_percentile` растёт при
ухудшении support; `support_percentile=100-distance_percentile` имеет обратный
смысл. Это эмпирические показатели локальной поддержки, не log-likelihood.

`evaluation.include_test=true` добавляет nearest test audit и изображения соседей.
Его оставляют выключенным при выборе параметров и включают после их фиксации.
Nearest train и test proximity не доказывают ни memorization, ни её отсутствие.
Train/validation indices evaluator и diffusion должны совпадать.

Stability: perturbation scale задан как RMS координаты, direction равномерно
генерируется из нормированного Gaussian vector. Сохраняются endpoints, return
rate, RMS distance до original и refined center, semantic consistency.
Curvature: центральная разность `[s(x+εv)-s(x-εv)]/(2ε)` для unit L2 vectors v,
без clipping точек разности. Значения, fraction_negative, mean/median/quantiles
сохраняются. Это не доказательство максимума: learned score может быть
неконсервативным, а результаты у границы box требуют отдельной интерпретации.

## PCA dimension pipeline

PCA fit использует только train. Сохраняются basis, mean, variance ratios,
исходные indices и transformed train/validation/test. Test только преобразуется,
не участвует в fit или обучении. Whitening выключен, масштаб коэффициентов не
подгоняется отдельно по d. MLP epsilon architecture соответствует стилю старого
векторного diffusion, но реализована отдельно. PCA basis включён в checkpoint.

Каждое d обучается отдельно. Decode кандидатов нужен только для pixel/evaluator
оценки и визуализации. Ограничивать коэффициенты [-1,1] некорректно, поэтому
PCA preset явно использует projection=none. Диапазон декодированных пикселей
диагностируется, а PNG для отображения ограничивается диапазоном.

**Ограничение dimension experiment:** усечение PCA меняет информацию и
распределение, не только ambient dimension. Сравнение pixel CNN с PCA MLP также
меняет архитектуру. Основная размерностная кривая сравнивает MLP между d;
PCA-784 служит контролем полного ортогонального представления. Сохраняются
retained variance и validation reconstruction RMS. Не интерпретируйте эту серию
как изолированное доказательство «проклятия размерности».

## Runner, presets, resume и файлы

YAML содержит `defaults`, `experiments` с `name,config,grid`, необязательный
`include` и `output_root`. Grid раскрывается декартовым произведением. Ключи
вида `search.alpha` меняют вложенное поле. Можно перебирать population, seeds,
checkpoint paths, все movement/merge settings, levels и final steps. В путях
доступны `{model_seed}`, `{dimension}`. `--only` выбирает group/job или glob.

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/ablation.yaml --only 'feature_geometry' --dry-run
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/main.yaml --execute --resume --device mps
```

Completed suite пропускается до загрузки моделей и данных. Interrupted suite
перезапускает незавершённый метод с теми же seeds, а completed methods пропускает.
Это resume серии/метода, не продолжение с произвольного внутреннего score step.
Обучение evaluator/PCA сохраняется на границах эпох. Изменившиеся checkpoint,
code hash или config блокируют повторное использование результата. Для другой
конфигурации выбирайте новый output root; не смешивайте версии реализации.

| Файл | Содержимое |
|---|---|
| `<job>/resolved_config.yaml`, `metadata.json` | Полный config, seeds, t/sigma, hashes, версии, device |
| `<stage>/checkpoints/<sha>.pt` | Неизменяемые копии diffusion/evaluator checkpoint |
| `<job>/<method>/arrays.pt` | Initial noise, populations, candidates, lineage, trajectories, endpoints |
| `<job>/<method>/levels.jsonl` | Все шаги, NFE, clipping, cluster statistics |
| `<job>/<method>/metrics.json`, `summary.json` | Initial/final metrics и компактные показатели |
| `<job>/<method>/*.png` | Initial/candidates, trajectories, train/test neighbors, mean/medoid |
| `complete.json`, `failure.json` | Статус возобновляемого запуска |

`torch.load(..., weights_only=True)` поддерживается для численных PT artifacts.
Значимые числа не хранятся только в PNG. Aggregation пишет JSON и CSV с individual
runs, mean/std/median/95% bootstrap CI и paired differences. Mean сбалансирован по
model seeds; bootstrap сначала выбирает модели, затем повторы внутри модели.
Один checkpoint даёт только conditional uncertainty. Paired difference всегда
`baseline minus original`, p-values и автоматического выбора «победителя» нет.
Condition IDs разделяют геометрию и другие параметры. Графики включают
quality/support/coverage/count/stability vs NFE, population, dimension и bubble
Pareto view с размером по числу кандидатов.

## Порядок будущих локальных запусков

Сначала смотрите `--dry-run` каждого этапа. Все команды ниже предполагают явное
решение запустить соответствующий этап; код здесь их не выполнял.

1. Basic diagnostic: COMMAND 1 выше. Проверьте артефакты и clipping/NFE.
2. Step/distance/merge diagnostics. Перед ними обучите один независимый evaluator:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/evaluator.yaml --execute --device mps
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/geometry.yaml --execute --device mps
```

3. Main baseline comparison на сохранённом original preset:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/main.yaml --execute --device mps
```

4. Population/NFE curves:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/matched_nfe.yaml --execute --device mps
```

5. Отдельные ablations, не объединяя pixel и feature результаты:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/ablation.yaml --execute --device mps
```

6. Dimension/PCA. Preparation включает одно PCA fit и семь обучений MLP:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/dimension_prepare.yaml --execute --device mps
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/dimension.yaml --execute --device mps
```

7. Multiple model seeds. Preparation обучает только новые CNN seeds 43/44:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/model_seeds_prepare.yaml --execute --device mps
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/model_seeds.yaml --execute --device mps
```

8. Финальный заранее зафиксированный запуск. Поставленный publication preset
сохраняет original geometry и включает test-neighbor audit. Если добавляется
обоснованный вариант, его config фиксируется до использования test:

```bash
python -m src.mnist_experiments.runner --matrix configs/mnist_experiments/publication.yaml --execute --device mps
python -m src.mnist_experiments.aggregate --root runs/mnist_experiments --output runs/mnist_experiments_aggregate
```

Существование этого кода и прохождение smoke tests не свидетельствуют об
эффективности v3f2 на MNIST. Такой вывод требует будущих локальных сравнений.
