# Выбор настроек и фиксация Confirmation v2

Исходные E0–E9, все новые Pilot jobs и перенос на старые модели43/44 —
exploration. Confirmation не запускается автоматически и не используется
для выбора шага, расписания, Merge radius или критериев принятия гипотезы.

## Настройка MNIST

Grid model42: L2 steps .01/.03/.06/.1; multi, short и single t=0/100/300/500.
Суммарно 45 main +10 final score calls на частицу. Merge mean radii
.1/.2/.35 RMS и medoid .2 — отдельные treatments. P0=128, search0–4.

Бюджет 13440 включает DDIM50 initialization и поиск. DDIM105 P128 использует
ровно 13440; DDIM50 expanded pool — 268 полных траекторий, 13400 queries.
Отдельный DDIM50 P128 контроль расходует 6400 и является cap-контролем.
Пулы и готовые starts не объявляются одинаковыми, если schedules различаются;
общим остаётся noise prefix. Native/final clustering оценивается общим правилом.

`propose-selection` читает **только** `mnist_tuning`, model42, native output,
пять завершённых допустимых search seeds. Model43/44 и Confirmation не могут
войти в этот расчёт. Данные должны относиться к текущим замороженным исходникам;
после изменения исходников нельзя использовать прежний tuning как незаметно
эквивалентный cohort.

Сначала применяется заранее заданный guard относительно DDIM105:

| Диагностика | Допустимая граница среднего |
|---|---|
| Class entropy | Снижение не более .2 |
| Pairwise RMS diversity | Не ниже .8 от DDIM105 |
| Near-duplicate fraction | Увеличение не более .05 |
| Pixel train-extreme-close fraction | Увеличение не более .05 |
| Feature train-extreme-close fraction | Увеличение не более .05 |

Train-extreme-close: nearest training distance ниже 1%-квантили validation→train
distance. Дополнительно сохраняются train/validation nearest ratio, точные
train copies, kNN support и neighbor IDs. Это диагностика необычной близости,
не доказательство memorization. Guards — консервативное правило отбора,
а не установленный научный закон; отклонение от них надо зафиксировать до
Confirmation с конкретным основанием.

Среди прошедших guard candidates строится Pareto по pixel/feature kNN внутри
семейств multi / single / short / merge. Нет произвольного weighted score.
Несколько nondominated настроек требуют явного выбора tradeoff в review notes.

Обязательные MNIST controls: `DDIM105`, `DDIM50_matched_pool`,
`multi_l2_003` и хотя бы один оптимизированный single. Всего выбирается 4–8
treatments. Для остальных кандидатов требуется guarded support frontier либо
явное научное обоснование в `manual_overrides`. Непрошедший guard вариант
можно фиксировать как отрицательный/механистический контроль, а не как
незаметно успешного победителя. Если выбрано несколько single treatments,
роль `mnist_roles.optimized_single` указывает главный comparator заранее.

## Фиксация

`selection_draft.json` содержит готовые операционные гипотезы и критерии:

- H_merge: под cap Merge снижает queries, сохраняя one-to-one recall в
  noninferiority margin .05, в high-dimensional close/rare strata.
- H_split: rare recall Split превосходит дополнительные независимые старты
  при matched queries, отдельно по размерности.
- H_mnist: current multi улучшает обе kNN distances относительно выбранного
  optimized single. Контраст определён как single−multi, поэтому положительная
  разность поддерживает multi. Diversity/proximity guards обязательны для
  содержательной интерпретации, но не объявляются отдельным доказательством
  train memorization.

Можно пересмотреть/сократить гипотезы **до lock** с явными review notes.
Каждый clause должен ссылаться на выбранный predeclared primary contrast,
metric, конечный threshold и направление. Невозможно зафиксировать H_split,
убрав Split из всех выбранных treatments. Синтетические controls M1, M2, M3,
M4, M7 остаются; M6/M8 выбираются по гипотезам. Их default steps фиксированы
в blueprint и должны иметь matching Pilot evidence.

`lock-confirmation` пересчитывает численные candidates из model42 и проверяет:

1. Выбор reviewable, с непустыми гипотезами и notes; guards/candidate table
   не изменены вручную.
2. Для выбранных treatments есть допустимые matching Pilot данные.
3. Confirmation distribution/search seeds отсутствуют в Pilot.
4. Входные модели и данные доступны; source/runtime/input SHA256 фиксируются.
5. Заданы непустые контрасты и операционные критерии гипотез.
6. В `results/confirmation/` ещё нет завершённых outcomes.

Сохранённый lock неизменяем: команда не перезаписывает существующий файл.
`run-confirmation` проверяет digest самого lock, исходники, версии Python/
пакетов, входные bytes и полный список jobs. Его нельзя подменить запуском
blueprint через `run-pilot`. Изменения кода/среды требуют отдельного явно
обозначенного протокола до результатов, а не исправления lock задним числом.

## Объём и область выводов

Подготовлены три blueprint-файла: exact matched, exact cap, MNIST. Полная
неотфильтрованная развёртка — 1740 jobs, но MNIST tuning grid целиком не
запускается: lock оставляет 4–8 настроек. При default семи exact treatments
получается 1480–1520 jobs, а не десятки тысяч.

Exact: новые distributions101–105, dimensions10/30/50, families close/rare/
sheared/collinear; search1000–1002. Cap: close/rare, M3/M4. MNIST: старые
models43/44, новые search1000–1004. Эти новые search seeds **не обеспечивают
независимость от уже известных обучений**. CI условны на фиксированный набор
моделей. Новые независимые обученные модели потребовали бы отдельной работы,
которую данный протокол не выполняет.

`analyze --lock ...` использует только зафиксированные contrasts/hypotheses.
`hypothesis_decisions.csv` применяет familywise CI к thresholds; неполный
план, отсутствующий CI или провал guard не даёт положительного решения.
Каждую размерность/семейство читать отдельно: успех в одном stratum не
становится общим утверждением о популяционном поиске.
