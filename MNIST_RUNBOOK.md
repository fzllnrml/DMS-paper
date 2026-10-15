# MNIST: первая unconditional diffusion model

Все команды выполняются из корня репозитория. Synthetic/Levine code и их
зависимости не изменены. MNIST не использует labels, PCA или алгоритмы поиска мод.

## Установка

Python 3.12, на Mac с Apple Silicon используйте ARM64 Python, без Rosetta.
Отдельное окружение не затрагивает окружение старых экспериментов:

```bash
python3 -m venv .venv-mnist
source .venv-mnist/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mnist.txt
```

`torch` и `torchvision` устанавливаются вместе как совместимая пара.
Для CUDA нужен PyTorch wheel под вашу установленную CUDA/драйвер.
`--device auto` выбирает CUDA, затем MPS, затем CPU. Явное недоступное устройство
вызывает ошибку. Mixed precision не используется, все вычисления float32.

## Проверка перед обучением

Быстрые offline tests, без скачивания MNIST:

```bash
python -m unittest discover -s tests -p 'test_mnist.py' -v
```

Smoke на настоящем MNIST: 32 train images, 16 validation images, 2 optimizer
steps, checkpoint/reload, DDIM из 4 шагов и batched score. Первый запуск
скачивает официальный MNIST целиком, но обучается только на указанном подмножестве.

```bash
python -m src.mnist.train --config configs/mnist.yaml --device mps --smoke
```

Для CPU замените `mps` на `cpu`. Чтобы повторить smoke, задайте новый
`--output-dir runs/mnist_smoke_2`. Smoke не перезаписывает основной запуск.

## Первое полноценное локальное обучение

```bash
python -m src.mnist.train --config configs/mnist.yaml --device mps
```

Конфигурация: небольшой U-Net 28 -> 14 -> 7, epsilon prediction, линейное
VP/DDPM расписание из 1000 шагов, MSE по шуму, AdamW, batch 64, 30 эпох.
Это начальная конфигурация, а не проверенный научный результат.
Split: 55 000 train / 5 000 validation из официального train set;
официальные 10 000 test images выделены отдельно и при обучении/выборе checkpoint
не используются. `uint8 / 127.5 - 1` переводит пиксели в [-1, 1].
Порядок split фиксирует `data.split_seed`, индексы сохраняются в checkpoint.
Validation loss усредняется по всем validation images с фиксированными t/noise.

## Resume

```bash
python -m src.mnist.train --config configs/mnist.yaml --device mps \
  --resume runs/mnist/checkpoints/latest.pt --epochs 30
```

`--epochs` задаёт общее число эпох, включая завершённые. Для продолжения после
30 эпох укажите, например, `--epochs 40`. Checkpoint сохраняется после каждой
полной эпохи. После Ctrl+C незавершённая эпоха будет выполнена заново.
Сохраняются модель, AdamW, номер эпохи/шага, конфигурация, split indices и RNG.
Начало нового запуска в существующем каталоге блокируется во избежание потери данных.
При resume модель, split, seed, batch size и настройки оптимизатора должны совпадать.
Можно изменить устройство, пути, частоту консольного вывода и целевое число эпох.
Для ответвления от `best.pt` задайте новый `--output-dir`.

При одинаковых seed, устройстве и версиях библиотек запуск воспроизводим.
Точное совпадение непрерывного обучения и resume проверяется unit test на CPU.
Между CPU/MPS/CUDA и версиями PyTorch побитовое совпадение не гарантируется.
Включён deterministic mode с предупреждением для неподдерживаемых операций.
Все случайные t/noise и shuffle используют отдельные сохраняемые CPU generators.

## Генерация DDIM

```bash
python -m src.mnist.sample --checkpoint runs/mnist/checkpoints/best.pt \
  --device mps --n 64 --steps 50 --seed 42 \
  --output runs/mnist/samples/ddim.png
```

DDIM использует eta=0 и отдельный seed, с последним переходом к чистому x0.
При генерации x0 ограничивается диапазоном [-1, 1]. В score clipping отсутствует.

## Файлы запуска

| Содержимое | Путь |
|---|---|
| MNIST cache | `data/mnist/MNIST/raw/` |
| Последняя полная эпоха / лучшая validation loss | `runs/mnist/checkpoints/latest.pt`, `best.pt` |
| Метрики каждой эпохи | `runs/mnist/logs/metrics.jsonl` |
| Фактическая конфигурация | `runs/mnist/config.yaml` |
| PNG с DDIM samples | `runs/mnist/samples/ddim.png` |
| Smoke checkpoint, logs, PNG | `runs/mnist_smoke/` |

Обучение не генерирует изображения автоматически. PNG создаёт отдельная команда
sampling, либо smoke. Кэш и результаты исключены из git.

## Контракт score для следующего этапа

```python
import torch
from src.mnist import MNISTDiffusion

model = MNISTDiffusion.from_checkpoint("runs/mnist/checkpoints/best.pt", device="mps")
x = torch.zeros(8, 784, dtype=torch.float32, device=model.device)
t = torch.tensor([0, 5, 15, 50, 100, 300, 600, 999], device=model.device)
scores = model.score(x, t)  # [8, 784], один вызов сети для всего батча
```

Поддерживаются `[B,1,28,28]`, `[B,784]` и одиночный `[784]`; выход сохраняет
форму, float32 и устройство входа. `t` задаётся целым scalar или integer tensor
`[B]` в диапазоне `[0,T-1]`. Это дискретный индекс, не непрерывное время.
Как в существующем `NoiseSchedule`, t=0 соответствует первому ненулевому шуму,
а не чистой плотности. `score = -epsilon / sqrt(1 - alpha_bar[t])` вычисляется
по нормализованным пикселям и не нормируется по длине. При переводе к сырым
пикселям [0,255] меняется масштаб score по правилу цепочки.

Доступны `dim=784`, `image_shape`, `device`, `schedule`, `score_numpy`,
`enable_nfe_counting/reset_nfe/nfe` (NFE считает изображения, не батчи),
`sample_ddim`. Score выполняется без autograd; для производных можно отдельно
использовать `model.net` и расписание. Режим train/eval после score восстанавливается.
Интерфейс подготовлен для будущей batched интеграции. Существующий v3f2 и его
DDIM helper не адаптированы: тот helper вызывает векторную сеть напрямую.
Для этой модели используйте `sample_ddim`, а не старый helper.
