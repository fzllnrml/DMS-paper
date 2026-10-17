# Зависимости от сохранённой первой версии

Патч не меняет ни одного файла вне `research_experiments_v2/`. Все перечисленные
зависимости доступны в исходном main commit
`10454bee79847a1246662fa5c280aba8b71b6801`.

| Сохранённый код | Как используется v2 | Почему не переписан |
|---|---|---|
| `research_experiments/common/storage.py` | Только чистые `clean`, `canonical`, `fingerprint`, `array_hash` | Каноническая сериализация и SHA256 массивов |
| `common/config.py` | `METHOD_DEFAULT`, строгие pure `merge` / `assign` | Совместимое отображение параметров математических kernels; не v1 loader/output paths |
| `common/cost.py` | `CountedScore`, `BudgetExhausted`, базовый `Cost` | Per-point counting, shape/finite validation, времена score |
| `common/search.py` | Только `move` | Проверенный шаг, clipping, displacement и движения |
| `common/geometry.py` | `cluster` complete-linkage и medoids | Геометрический output без доступа к эталону |
| `common/distributions.py` | `GMM`, `Schedule`, `ExactField`, `generate(legacy_random)`, incomplete fallback `reference_modes` | Точная VP GMM и прежний RNG порядок checkpoints |
| `common/historical.py` | `LegacyProxy`, `get_class`, `run_legacy` | Реальные сохранённые классы; reset_nfe не стирает внешний ledger |
| `common/models.py` | `LearnedField` | Только загрузка существующей архитектуры/весов, без обучения |
| `common/metrics.py` | `evaluate_modes`, `pareto_mask` | Максимальное one-to-one matching, затем минимальная дистанция; геометрический Pareto |
| `src/algorithms/mode_finder_v2.py`, `mode_finder_v3_f1.py`, `mode_finder_v3_f2.py` | Оригинальные методы через адаптеры | Сохранены arithmetic, stopping rules и полные control treatments |
| `src/mnist_experiments/evaluator.py`, `data.py`, `evaluation.py` | FrozenEvaluator, ReferenceData, nearest_neighbors, semantic_metrics, support_percentiles | Те же preprocessing, frozen features, direct-distance kNN и train IDs |
| `src/models/`, `src/mnist/model.py` | Косвенно через load-only adapters | Исходные архитектуры checkpoints и преобразование epsilon→score |

`common` в таблице — подкаталог **первой** `research_experiments/`.
V2 **не вызывает** v1 runners, `JobStore`, `safe_path`, `initialize`,
`evaluate_mnist`, training scripts или v1 аналитические writers. Их пути записи
остаются ориентированными на первую серию, поэтому новые storage/cache/runner
не могут быть обёрткой над этими writers.

Все Python/JSON/Markdown/requirements исходники v2, v1 и `src` хэшируются в
source identity. Настоящие версии пакетов и конкретные input SHA256 входят в
metadata; Confirmation проверяет их до исполнения. Локальное `.venv`, результаты,
датасеты и кэши не входят в source digest. Импорты v2 отключают запись bytecode
в сохранённые директории.

## Изменения поведения, сделанные только в v2

1. **Output constraints.** В первой серии возможен `requested_k=8` при выходе
   двух кандидатов. V2 различает exact K, ≤K и native output; не дополняет
   недостающие кандидаты копиями. Невыполненный exact K явно невалиден.
2. **Бюджет.** Общий ledger включает initialization, независимые старты, FD
   Hessian и probes. Exact/learned pair с neural initialization не может
   обойти лимит аналитическими вызовами другого типа. Счётчики типов отдельны.
3. **Прерванные рестарты.** В v1 `run_search` новый anchor добавляется до
   refinement; при исчерпании бюджета он может остаться без кандидата. V2
   заранее резервирует полную стоимость DDIM+плана и коммитит anchor с готовым
   потомком. Не тратит query на рестарт, который заведомо не помещается.
4. **Split.** Родитель заменяется только завершённой интервенцией. Стоимость
   прерванной Hessian/probe попытки сохраняется; незавершённые дети не
   материализуются. Глобальный population cap координатора отличается от
   исторического per-start cap и явно описан в параметрах.
5. **Stable Merge.** Малый последний clipped step не является доказательством
   устойчивости. V2 требует окно малых фактических перемещений и платную
   проверку score norm; direction gate тоже оплачивается.
6. **v3f1 softness.** В оригинальном `_select_split_directions` при всех
   отрицательных eigenvalues отношение `lambda / abs(lambda_min)` отрицательно,
   но сравнивается с положительным `softness_threshold`. Эта ветвь не срабатывает.
   Оригинальный kernel остаётся отдельным treatment. Новый
   `v3f1_softness_v2` меняет только эту ветвь, определяя softness как
   `1 - abs(lambda)/max(abs(negative_lambda))` и выбирая наиболее мягкое
   направление выше threshold. Это **явно новая абляция**, а не тайное
   исправление исторического v3f1 и не доказательство его авторского замысла.
7. **Ground truth.** V1 numerical multistart не превращается в completeness
   proof. V2 добавляет специальные доказуемые семейства и outward IV
   certificates; произвольная GMM остаётся incomplete. Неполные denominators
   маскируются в основных recall/F1 полях.
8. **DDIM streams.** Малый адаптер сохраняет исходные update formulas, но
   принимает explicit noise и logical start indices. Это необходимо для
   сопоставления больших пулов и поздних стартов. Старые torch-seed samples
   этим не объявляются воспроизведёнными.
9. **Статистика.** Есть ненулевой список заранее заданных контрастов,
   treatment-level strata, парные разности, cluster-aware CI, fixed-model MNIST
   scope, multiplicity по плану и аудит недостающих jobs. Дубликаты настроек
   при pairing вызывают ошибку, не скрытое усреднение.
10. **MNIST.** Независимый writer/cache сохраняет прежние kNN definitions и
    добавляет training-proximity flags, validation proximity, displacement,
    ancestry/class changes и diversity guards. Присвоение класса не
    интерпретируется как обнаружение истинной моды.

Окружение исполнения Python не является полностью независимым от v1:
удаление или изменение перечисленных исходников нарушает зависимость и
инвалидирует lock. Независимы новые протоколы, writers, результаты, настройки,
собственные исправления и жизненный цикл второй серии.
