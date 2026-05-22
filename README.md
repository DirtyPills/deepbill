# OpenAI Adapter Deep Bill

## 📋 Обзор проекта

**Deep Bill** — это комплексное решение для эмуляции OpenAPI за счет бесплатного чата deep seak.

- Эмулировать поведение OpenAI API
- Обрабатывать вызовы функций (function/tool calling)
- Интегрироваться с DeepSeek runtime
- Проводить валидацию адаптеров и производительности

## 🎯 Цели проекта

1. **Эмуляция OpenAI API** — предоставить полностью совместимый адаптер для имитации работы OpenAI API
2. **Tool Calling** — реализовать парсинг и выполнение вызовов функций
3. **Тестирование маршрутов** — создать инструменты для live-тестирования API-эндпоинтов
4. **Интеграция моделей** — обеспечить совместимость с DeepSeek и другими runtime
5. **Кросс-платформенность** — поддерживать Linux и Windows

## 🏗 Архитектура проекта

### Основные компоненты

| Файл | Назначение | Ключевые функции |
|------|------------|------------------|
| `openai_adapter.py` | Основной адаптер OpenAI API | Эмуляция запросов, обработка ответов, совместимость с OpenAI SDK |
| `app.py` | Главный сервер приложения | Маршрутизация запросов, управление сессиями, логирование |
| `deepseek_runtime.py` | Интеграция с DeepSeek | Выполнение инференса моделей DeepSeek, управление контекстом |
| `tool_call_parser.py` | Парсер tool-вызовов | Разбор JSON-схем функций, валидация параметров, выполнение |
| `live_continue_route_test.py` | Тестирование маршрутов | Live-мониторинг эндпоинтов, проверка доступности, замеры времени |
| `adapter_route_tests.py` | Валидация адаптера | Юнит-тесты, интеграционные тесты, проверка совместимости |
| `runtime_tests.py` | Тесты производительности | Бенчмарки, нагрузочное тестирование, профилирование |

## 🔧 Установка и настройка

### Требования к системе

- **Python**: 3.8 или выше
- **ОС**: Linux (Ubuntu 20.04+) или Windows 10/11
- **Оперативная память**: минимум 2GB (рекомендуется 4GB+)
- **Дисковое пространство**: минимум 500MB

### Установка зависимостей

```bash
# Установка через pip
pip install -r requirements.txt
```

Содержимое `requirements.txt`:
```
openai>=1.0.0
flask>=2.0.0
requests>=2.25.0
pyyaml>=5.4.0
```

### Настройка конфигурации

Отредактируйте файл `dbill_settings.json`:

```json
{
  "api_endpoint": "http://localhost:5000",
  "model_name": "Deep Bill AI",
  "max_tokens": 128000,
  "temperature": 0.7,
  "routing_rules": {
    "default": "openai-adapter",
    "deepseek": "deepseek-runtime"
  }
}
```

## 🚀 Запуск

### Linux / macOS

```bash
# Дать права на выполнение
chmod +x run_linux.sh

# Запустить
./run_linux.sh
```

### Windows

```cmd
run_windows.bat
```

### Ручной запуск

```bash
# Запуск адаптера
python openai_adapter.py

# Запуск сервера
python app.py

# Запуск с конкретной конфигурацией
python app.py --config continue-deepbill.config.yaml
```

## 🧪 Тестирование

### Запуск всех тестов

```bash
# Тесты адаптера
python adapter_route_tests.py

# Live-тесты маршрутов
python live_continue_route_test.py

# Тесты парсера tool-вызовов
python tool_call_parser_tests.py

# Тесты производительности
python runtime_tests.py
```

### Ожидаемые результаты

- `adapter_route_tests.py` — вывод статуса PASS/FAIL для каждого теста
- `live_continue_route_test.py` — отчёт о доступности эндпоинтов
- `runtime_tests.py` — бенчмарки времени выполнения

## 📊 Использование

### Базовый пример

```python
from openai_adapter import OpenAIAdapter

# Создание адаптера
adapter = OpenAIAdapter(config_path="dbill_settings.json")

# Отправка запроса
response = adapter.chat_completion(
    messages=[{"role": "user", "content": "Привет, мир!"}],
    model="gpt-3.5-turbo",
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
```

### Пример с tool calling

```python
from tool_call_parser import ToolCallParser

# Определение функции
functions = [
    {
        "name": "get_weather",
        "description": "Получить погоду в городе",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }
]

# Парсинг вызова
parser = ToolCallParser(functions)
result = parser.parse_and_execute('{"name": "get_weather", "arguments": {"city": "Moscow"}}')
```

### Пример тестирования маршрута

```python
from live_continue_route_test import RouteTester

tester = RouteTester(config="dbill_settings.json")
status = tester.check_route("/v1/chat/completions", method="POST")
print(f"Статус: {status}")
```

## 📁 Структура проекта

```
.
├── adapter_route_tests.py        # Тесты маршрутизации адаптера
├── adapter_test_run.log           # Лог тестов
├── app.py                         # Главный сервер
├── continue-deepbill.config.yaml  # Конфигурация Continue + DeepBill
├── dbill_settings.json            # Основные настройки
├── deepseek_profile/              # Профили DeepSeek
├── deepseek_runtime.py            # DeepSeek runtime
├── install.sh                     # Установщик для Linux
├── install_windows.bat            # Установщик для Windows
├── live_continue_route_test.py    # Live-тесты маршрутов
├── live_continue_route_test.log   # Лог live-тестов
├── nohup.out                      # Вывод фонового запуска
├── openai_adapter.py              # Основной адаптер OpenAI
├── OPENAI_ADAPTER_GUIDE.md        # Руководство по адаптеру
├── __pycache__/                   # Кэш байт-кода
├── README.md                      # Основной README
├── requirements.txt               # Зависимости Python
├── run_linux.sh                   # Скрипт запуска для Linux
├── run_windows.bat                # Скрипт запуска для Windows
├── runtime_tests.py               # Тесты производительности
├── TESTING.md                     # Руководство по тестированию
├── tool_call_parser.py            # Парсер tool-вызовов
└── tool_call_parser_tests.py      # Тесты парсера
```

## 🔍 Логирование и отладка

### Файлы логов

- `adapter_test_run.log` — результаты тестирования адаптера
- `live_continue_route_test.log` — результаты live-тестов
- `nohup.out` — вывод при фоновом запуске

### Уровни логирования

По умолчанию используется уровень INFO. Для отладки измените конфигурацию:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🐛 Известные проблемы и решения

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'openai'` | Установите: `pip install openai` |
| Ошибка подключения к API | Проверьте `api_endpoint` в `dbill_settings.json` |
| Tool call не выполняется | Убедитесь, что функция определена в JSON-схеме |
| Высокая задержка ответа | Запустите `runtime_tests.py` для бенчмарка |

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для фичи: `git checkout -b feature/amazing-feature`
3. Зафиксируйте изменения: `git commit -m 'Add amazing feature'`
4. Запушьте: `git push origin feature/amazing-feature`
5. Откройте Pull Request

## 📄 Лицензия

MIT License — подробности в файле LICENSE (если применимо).

## 📞 Контакты и поддержка

- **Автор**: duke16bit
- **GitHub**: [github.com/duke16bit](https://github.com/duke16bit)
- **Email**: duke16bit@gmail.com

## 🙏 Благодарности

- OpenAI за спецификацию API
- DeepSeek за runtime интеграцию
- Сообществу open-source за инструменты тестирования

---

**⭐ Если проект вам полезен, поставьте звезду на GitHub!**

