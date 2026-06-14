# DBillTOGIT

`DBillTOGIT` - это часть проекта DEEPBILL, отвечающая за чат и
OpenAI-совместимый адаптер. В этой копии находятся браузерная автоматизация
DeepSeek, ручной GUI чата, адаптер OpenAI с поддержкой инструментов,
управление состоянием адаптера и примеры клиентской конфигурации. Захват
скриншотов, горячие клавиши экрана, запись голоса, экранные доки и ассеты
скриншотов в эту копию не входят.

## Что Запускается

- `app.py` запускает Tk GUI и постоянный браузер Playwright с DeepSeek.
- `deepseek_runtime.py` отправляет текстовые промпты, удаляет элементы
  управления DeepSeek из собранного текста ответа, автоматически нажимает
  видимые кнопки `Continue` и объединяет части продолжения.
- `openai_adapter.py` предоставляет Chat Completions и legacy completions для
  Roo Code, Continue и других OpenAI-совместимых клиентов.
- `tool_call_parser.py` преобразует текст tool-call из веб-интерфейса в
  нативные OpenAI `tool_calls`.
- `scripts/dbillctl.sh` управляет тихим headless-сервисом адаптера, который
  используется shell-алиасами `start_db`, `stop_db` и `status_db`.

Структура проекта:

- `tests/` - детерминированные и live-проверки маршрутов.
- `docs/` - заметки по адаптеру, тестированию и планам реализации.
- `config/` - примеры клиентской конфигурации.
- `logs/` - локальные логи запусков и тестов.
- `runtime/` - локальные pid/state-файлы для тихого управления сервисом.

## Установка

Linux:

```bash
chmod +x install.sh run_linux.sh
./install.sh
```

Windows:

```bat
install_windows.bat
```

Установщик создает `.venv`, устанавливает Python-зависимости и скачивает
браузер Playwright Chromium.

## Запуск

Linux:

```bash
./run_linux.sh
```

Windows:

```bat
run_windows.bat
```

Первый запуск открывает веб-страницу DeepSeek через постоянный профиль
браузера. Если DeepSeek попросит авторизацию, войдите в аккаунт в этом окне.
В GUI есть кнопка `Open Browser`, которая снова выводит вкладку DeepSeek на
передний план.

Локальная директория `deepseek_profile/` создается автоматически при первом
запуске. Она хранит браузерную авторизацию только на машине пользователя и
исключена из Git.

Тихий headless-сервис адаптера:

```bash
start_db
status_db
stop_db
```

Эти алиасы устанавливаются в `~/.bash_aliases` и используют тот же постоянный
профиль DeepSeek. Они не открывают Tk GUI. Десктопный лаунчер по-прежнему
вызывает `run_linux.sh`, поэтому запуск кликом остается путем с видимыми GUI и
браузером.

## Адаптер

Запустите адаптер из GUI. Базовый URL по умолчанию:

```text
http://127.0.0.1:8080/v1
```

В OpenAI-совместимом клиенте используйте модель `deepseek-chat` и любой
непустой API-ключ. Адаптер поддерживает `/v1/models`,
`/v1/chat/completions`, `/chat/completions`, `/v1/completions`,
SSE-ответы и нативные OpenAI tool-call ответы.

Для крупных агентских задач у адаптера явно задано поведение backpressure и
восстановления:

- одновременно выполняется один активный браузерный запрос DeepSeek;
- небольшая ограниченная очередь ожидания через `DEEPBILL_ADAPTER_QUEUE_LIMIT`;
- `429 server_busy`, когда очередь заполнена или браузер остается занятым
  дольше `DEEPBILL_ADAPTER_BUSY_TIMEOUT`;
- `503 server_unavailable` с `Retry-After`, когда circuit breaker открывается
  после повторяющихся runtime-сбоев.

Значения по умолчанию для ответов подготовлены под большие ответы:

- Таймаут ответа браузера по умолчанию: `360` секунд.
- Ожидание завершения ответа DeepSeek по умолчанию: `2.5` секунды. Если при
  медленном соединении между потоковыми частями ответа браузера возникают
  паузы, измените `Finish wait, sec` в GUI.
- Таймаут адаптера по умолчанию: `360` секунд через
  `DEEPBILL_ADAPTER_TIMEOUT`.
- Верхняя граница принимаемого таймаута адаптера по умолчанию: `1800` секунд
  через `DEEPBILL_ADAPTER_MAX_TIMEOUT`.

## Проверка

Детерминированные проверки:

```bash
.venv/bin/python tests/runtime_tests.py
.venv/bin/python tests/tool_call_parser_tests.py
.venv/bin/python tests/adapter_route_tests.py
.venv/bin/python tests/roocode_simulation_tests.py
```

Для live-проверок в стиле Continue нужен запущенный GUI-адаптер с выполненной
авторизацией:

```bash
.venv/bin/python tests/live_continue_route_test.py --base-url http://127.0.0.1:8080/v1 --timeout 360
```

`tests/live_continue_route_test.py` запускает небольшой безопасный локальный
tool sandbox, проверяет агентские сценарии create, read, edit, read через
OpenAI-маршрут и повторяет крупный запрос на монолитный код, отслеживая
runtime-счетчик `Continue`. Добавьте `--require-continuation`, если
стресс-проверка больших ответов должна падать, когда DeepSeek отклоняет каждый
крупный запрос кода.
