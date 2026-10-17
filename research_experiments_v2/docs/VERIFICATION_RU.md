# Проверка патча и сохранности первой серии

База: main `10454bee79847a1246662fa5c280aba8b71b6801`.
До реализации снят SHA256 inventory всех 1853 существующих файлов свежего
checkout, исключая служебную `.git`. Существующих files вне v2 не редактировали.

## Проверки к сдаче

- Все добавления находятся внутри `research_experiments_v2/`; патч не содержит
  modified/deleted/renamed прежних путей.
- `git apply --check` и реальное применение выполнены на отдельном чистом
  checkout этой базы. SHA256 всех 1853 прежних файлов до/после совпадают;
  это включает `research_experiments/`, `src/`, `checkpoints/` и прежние
  tracked результаты. В исходном рабочем checkout inventory также совпадает.
- Unit suite: 49 тестов, только малые искусственные функции/массивы и
  разбор неизменённых legacy class bodies. В частности проверяются original
  v3f2 parity на 2D fixture, стоимость FD2d, частичный Split, общий стартовый
  stream, транзакционные рестарты, K constraints, metadata/resume/hashes,
  невозможность запуска Confirmation без lock, изменение locked input bytes,
  fixed-model uncertainty и incomplete-reference masking.
- Проверена развёртка всех Pilot и Confirmation JSON: 4408 Pilot jobs,
  1740 потенциальных blueprint jobs до отбора. Ни одного score call или
  checkpoint load при dry-run.
- Проверен анализ пустого каталога: 75 объявленных contrast-status rows,
  0 estimates, 0 научных результатов; отсутствие данных не выдаётся за эффект.
- Выполнен read-only аудит присланного ZIP: 204 jobs, 789 metric rows,
  575 historical rows, пустые requested contrasts, CRC без ошибок.
- AST/syntax и import checks не вызывают training/inference. `.gitignore`
  исключает results/env/cache/dataset/model/temp файлы.

## Что не проверялось экспериментальным запуском

Torch/torchvision, MNIST dataset и Mac MPS в окружении подготовки отсутствуют.
Сохранённые MNIST checkpoints/evaluator присутствуют и не изменены.
Нейросетевой forward, training, Pilot и
Confirmation не выполнялись. Не получено новых научных метрик. Сохранённые
checkpoints не переобучались и не перезаписывались. Локальный запуск на
MacBook должен начинаться с `doctor --inspect-checkpoints`.

## Повторение проверки пользователем

```bash
python -m unittest discover -s research_experiments_v2/tests -v
python -m research_experiments_v2 verify-isolation
python -m research_experiments_v2 dry-run
```

`verify-isolation` сравнивает literal file bytes с указанным Git commit,
без git content filters. Две checkpoint aliases проверяются как символические
ссылки; их tracked targets проверяются отдельно по байтам.
Для собственных незакоммиченных старых результатов
предусмотрен отдельный read-only `snapshot-isolation` до запуска и
`verify-isolation --baseline ...` после. Write guards v2 запрещают запись
в первую серию и не используют её результаты как новые v2 jobs.
