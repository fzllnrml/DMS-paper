# Совместимость сохранённых моделей

Ни одна команда v2 не обучает, не дообучает и не скачивает модель. Отсутствующие
файлы означают отказ preflight выбранной серии. Primary exact benchmark вообще
не нуждается в нейросетевых checkpoints.

## MNIST

Проверяются `mnist_diffusion_v1`, `config.seed`, `model_config` с T/beta endpoints,
`split_indices`, архитектура при загрузке и frozen evaluator format. Индексы
train/validation должны в точности совпадать у diffusion и evaluator.
Координаты — flatten изображений 28×28 в [-1,1]. Score вычисляется исходной
моделью, без переопределения её архитектуры или шкалы времени. Физическое
устройство и hash реального alpha_bar записываются.

## Синтетические checkpoints

Старый `DiffusionModel.save_checkpoint` сохраняет dim, T, hidden_dims и веса,
но **не сохраняет** target GMM, beta endpoints, activation, координатную
нормировку и prediction target. Имя файла само по себе не доказывает их
совместимость. Поэтому v2:

1. Загружает существующие веса, проверяет dim/T и явно заданную архитектуру.
2. Восстанавливает именно `legacy_random` с исходным RNG-порядком.
3. Передаёт exact полю реальное float32 alpha_bar обученной модели.
4. Использует learned DDIM для одинакового начального prefix exact/learned.
5. После поиска проверяет в одинаковых координатах направление, нормы,
   absolute/relative score error, near-reference probes и numerical basin endpoints.
6. Отдельным, **оплаченным диагностическим** forward проверяет численную
   identity `score = -epsilon / sqrt(1-alpha_bar_t)`.
7. При отсутствии исходных training evidence ставит
   `compatibility_verified=false`: условные разности остаются в диагностике,
   но не в подтверждающем ranking. Даже успешная epsilon identity не доказывает,
   что checkpoint обучался на ожидаемой GMM.

После диагностического запуска в `metadata.json → compatibility.expected`
есть точные ожидаемые поля. Для подтверждения нужен внешний JSON, переданный
как `--compatibility-manifest /path/to/evidence.json`:

```json
{
  "checkpoints": {
    "ACTUAL_CHECKPOINT_SHA256": {
      "density_sha256": "ACTUAL_EXPECTED_DENSITY_SHA256",
      "coordinates": "original_x",
      "beta_start": 0.0001,
      "beta_end": 0.02,
      "T": 1000,
      "activation": "silu",
      "prediction_target": "epsilon",
      "score_conversion": "-epsilon/sqrt(1-alpha_bar_t)",
      "evidence": "Путь/идентификатор исходного training config и журнала; способ сверки target parameters"
    }
  }
}
```

Пустая строка evidence не считается проверкой. Противоречащие поля приводят к
ошибке. Не заполнять эту таблицу догадкой по имени checkpoint. Наличие такого
JSON фиксирует проверенное пользователем происхождение, а не создаёт его.
Изменение evidence после завершённого v2 job меняет его identity: для новой
проверки использовать новый `--run-root`, не перезаписывать старый результат.

## Clean и indexed zero

Primary exact: t=0 соответствует alpha=1 и чистому эталону. Alignment/MNIST:
t=0 — сохранённый индекс модели с alpha≈0.9999, не alpha=1. При alignment
coverage оценивается относительно clean GMM, а поле на t=0 остаётся indexed.
Residual near clean mode поэтому не обязан быть нулевым даже у exact поля.
Это известная часть протокола, а не автоматически ошибка обучения.

Проверка совместимости — необходимое условие интерпретации. Она не превращает
ошибки направлений, расписания, clipping, budget allocation и конечного
числа шагов в единую «ошибку нейросети».
