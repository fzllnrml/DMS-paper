# Технический отчёт: DMS experiments v2

## Результат разработки

Подготовлен добавочный патч к актуальному main
`10454bee79847a1246662fa5c280aba8b71b6801`. Он добавляет только
`research_experiments_v2/`. Ни одного изменения `research_experiments/`,
`src/`, `checkpoints/`, сохранённых результатов и других прежних файлов
не требуется. Root `.gitignore` также не изменяется: правила находятся
внутри новой директории.

V1 E0–E9 остаётся воспроизводимым в прежнем виде. V2 импортирует её чистые
kernels, но имеет отдельные протоколы, storage/guards, конфигурации,
исправленные варианты, статистику и результаты. Полного копирования старого
движка нет. [Зависимости и отличия](DEPENDENCIES_AND_CHANGES_RU.md).

## Добавленные файлы

Все пути ниже относительны `research_experiments_v2/`.

| Файлы | Назначение |
|---|---|
| `__init__.py`, `__main__.py` | Import без запуска; CLI doctor/dry-run/Pilot/analyze/selection/lock/Confirmation/audit/isolation |
| `storage.py` | V2-only path guard с resolve symlinks; atomic writes, exclusive lock creation, hashes, completion manifests |
| `config.py` | Строгая схема, Cartesian expansion, полные treatment signatures, внешние input paths |
| `cost.py` | Наследник прежнего Cost: жёсткий общий query cap, отдельные типы и component scopes |
| `initialization.py` | Малый DDIM adapter: explicit logical noise stream, общий P0 batch, v2 caches с intrinsic/executed cost |
| `engine.py` | Координатор бюджетов/P/K, транзакционные Split/restarts, сохранение событий; вызывает v1 move/geometry |
| `adapters.py` | Оригинальные классы/Split kernels, Hessian instrumentation; отдельная softness-v2 абляция |
| `problems.py` | Exact product/GMM densities, структурные mode proofs, outward IV certificates, special isotropic-pair completeness |
| `evaluation.py` | Post-search reference matching, incomplete masking, cross-basin Merge/Split diagnostics, exact/learned alignment |
| `mnist.py` | Frozen evaluator и прежние kNN definitions; v2 reference cache, class/ancestry/diversity/train-close diagnostics |
| `runner.py` | Общий preflight, загрузка готовых моделей, job identities, resume, failure ledger, только v2 outputs |
| `analysis.py` | Непустые заданные contrasts, hierarchical/fixed-model CI, planned multiplicity, paired/design audits, balanced Pareto, PNG plots, hypothesis decisions |
| `protocol.py` | Model42-only guarded Pareto selection; reviewable hypotheses; immutable Confirmation-lock и его проверка |
| `archive_audit.py` | Отдельный read-only аудит ZIP первой серии и точечный экспорт первичных данных |
| `isolation.py` | Побайтовое сравнение tracked старых файлов с commit; дополнительный snapshot локальных старых результатов |
| `configs/01_...` — `10_....pilot.json`, `pilot_all.json` | Полная новая серия 4408 jobs, отдельные exact/matched/cap/ablation/population/original/alignment/MNIST studies |
| `configs/confirmation_exact.blueprint.json`, `confirmation_exact_cap.blueprint.json`, `confirmation_mnist.blueprint.json`, `confirmation_all.json` | Новые seeds, заранее определённые контрасты; запуск только через lock |
| `configs/external_paths.example.json` | Карта произвольных внешних моделей/evaluator без требования одной структуры каталогов |
| `tests/__init__.py`, `tests/test_v2.py` | Малые математические, бюджетные, persistence, statistical/protocol fixtures; без checkpoint inference |
| `.gitignore`, `requirements.txt`, `README.md` | Изоляция генерируемых файлов, отдельная среда, точные команды MacBook |
| `docs/PROTOCOL_RU.md`, `EXPERIMENT_MATRIX_RU.md` | Полный научный план, оси P/K/NFE, метрики, гипотезы и ограничения |
| `docs/SELECTION_AND_CONFIRMATION_RU.md`, `CHECKPOINT_COMPATIBILITY_RU.md` | Настройка/подтверждение, fixed-model scope, недостающая training provenance |
| `docs/V1_ARCHIVE_AUDIT_RU.md`, `v1_evidence_request.json` | Проверенная достаточность ZIP, конкретные недостающие job/files, read-only экспорт |
| `docs/DEPENDENCIES_AND_CHANGES_RU.md`, `VERIFICATION_RU.md`, этот отчёт | Контракт с v1, явные исправления, фактические проверки |

## Какие прежние проблемы закрыты

1. Пустая таблица contrasts заменена явно заданным планом. Даже до запуска
   появляется статус «нет завершённых пар», а не фиктивный estimate.
2. Поиск с полным и неполным reference не смешивается: неполный denominator
   не может попасть в full recall ranking.
3. `label` не агрегирует разные настройки. Каждая комбинация шага/уровней/
   Merge/Split/initialization имеет собственную сигнатуру.
4. Cap, matched queries, strict K и ≤K разделены, с явной eligibility.
5. Hessian finite differences, платные gates, probes и дополнительные starts
   расходуют общий лимит. Cached initialization не исчезает из intrinsic NFE.
6. Старый риск dangling restart anchor устранён новым транзакционным
   координатором. Неполный Split не заменяет родителя.
7. Обнаруженная отрицательная softness-ветвь v3f1 сохранена как historical
   control; новый вариант назван отдельно и тестируется на той же арифметике.
8. Добавлены действительно высокомерные задачи с разным числом/редкостью мод,
   фиксированным сигналом при смене d и доказуемым полным mode catalog.
9. MNIST P16/один seed заменён планом P128/search0–4; model42 tune отделён
   от fixed exploratory transfer43/44 и будущей Confirmation.
10. Улучшение kNN сопровождается class/diversity/training-proximity audit;
    коллапс не объявляется множественным поиском мод.
11. Exact/learned uses common starts и actual schedule; недостающая training
    metadata остаётся явно неизвестной, а не восстанавливается предположением.
12. Aggregate Pareto сравнивает общие допустимые задачи, а design audit
    показывает недостающие planned jobs. Частицы/радиусы не увеличивают
    число независимых распределений/обученных моделей.

## Практические ограничения

- Все новые серии **ещё не выполнены**. Не установлены победитель, полезность
  Split/Merge или новизна для статьи. Это результат будущих вычислений.
- Архив первой серии достаточен для планирования, но не содержит первичных
  траекторий; их нельзя реконструировать из scalar CSV.
- Доказательство полноты относится к специальным конструкциям/редукциям.
  Обобщённая arbitrary-GMM completeness задача не объявляется решённой.
- Для shear известны моды, но не точные Euclidean basin probabilities.
- В старых синтетических checkpoint отсутствует часть training metadata;
  без внешней сверки alignment остаётся условной диагностикой.
- Существующие MNIST checkpoints и evaluator присутствуют в checkout и
  сохранены. В окружении нет torch/torchvision, MNIST dataset и Mac MPS.
  Проверки pretrained model inference и MPS не выполнялись. Их совместимость
  проверяет локальный `doctor --inspect-checkpoints` перед пользовательским
  запуском; unit parity использует неизменённые NumPy class bodies.
- Допустимые dependency ranges не заменяют фиксацию реальных версий;
  Confirmation-lock хранит именно использованные версии.
- Небольшое число независимых distribution seeds ограничивает точность CI;
  100000 bootstrap draws не являются 100000 экспериментами. MNIST CI
  условны на сохранённые models43/44, а не на новые trainings.
- Полный Pilot не обязан закончиться за одну ночь. Число queries не переводится
  в обещание времени; resume позволяет продолжать ту же серию без пересчёта
  готовых jobs. Для wall time нужен отдельный запуск без initialization cache.

## Передача

Передаётся `.patch`, применимый через `git apply`, и документация. Полные
results, dataset, `.venv`, новые weights и вычислительные кэши не входят в патч.
Git commit/push, обучение и полномасштабные запуски не выполнялись.
Проверки применения и побайтовой сохранности описаны отдельно в
[VERIFICATION_RU.md](VERIFICATION_RU.md).
