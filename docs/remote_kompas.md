# Работа с КОМПАС на Windows с Mac

## Кратко

Codex работает на Mac. КОМПАС-3D и portable MCP `ksapi-expert` находятся на Windows,
доступном по SSH-алиасу `windows_server` из `~/.ssh/config`.

Изолированные эксперименты и ограниченный `kompas-builder` находятся в ветке
[`ml-rnd`](https://github.com/BigMak1/kompas_auto_draft/tree/ml-rnd).
Прямой SSH из следующих разделов предназначен для оператора, не для построителя benchmark.

## Инструменты

- **MCP**: поиск документации KsAPI, чтение фрагментов, поиск точных имён в Python-обёртках
  и статическая проверка кода. MCP сам не выполняет код в КОМПАС.
- **SSH + Python на Windows**: выполнение скриптов KsAPI в живом КОМПАС. Для подключения
  используется `ksapi.GetKompas()`, поэтому КОМПАС нужно предварительно открыть.

Рекомендуемый цикл: `search` → `fetch` → `grep_ksapi_wrappers` → `validate_code`
с `context="standalone_python"` → выполнение через SSH → проверка результата в КОМПАС.
Статическая проверка не подтверждает правильность геометрии.

## Пути текущего окружения

| Что | Путь на Windows |
|---|---|
| Portable MCP с Python и моделью | `C:\MCP\ksapi-expert-mcp` |
| Python для скриптов и MCP | `C:\MCP\ksapi-expert-mcp\python\python.exe` |
| Обёртки установленного КОМПАС | `C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin` |
| Общая с Mac папка проекта | `C:\Mac\Home\Documents\VS_code_projects\ITMO\ASCON\kompas_auto_draft` |

Файлы проекта доступны Windows через общую папку Parallels. После изменения скрипта
на Mac копировать его на Windows не требуется. Portable MCP находится на локальном
диске Windows; локальная папка `mcp/` содержит исходную поставку и
исключена из Git.

## Подключение MCP

`scripts/ksapi_mcp_ssh.sh` запускается на Mac и открывает SSH без псевдотерминала.
На Windows он запускает `scripts/ksapi_mcp_stdio.py` встроенным Python из portable MCP.
Тот вызывает штатный `launch` из `ksapi_expert_mcp.deployment.cli`, используя модель,
индекс, обёртки и ограничения из `C:\MCP\ksapi-expert-mcp\config\server.json`.

MCP передаёт JSON-RPC по stdin/stdout через SSH. Отдельный HTTP-сервер, порт 18843
и постоянно работающий туннель для этого подключения не нужны. Сервер запускается
при подключении клиента и завершается при закрытии stdin. В stdout нельзя добавлять
баннеры или диагностические сообщения.

Для Codex конфигурация находится в `.codex/config.toml` текущего доверенного проекта:

```toml
[mcp_servers.ksapi-expert]
command = "/bin/bash"
args = ["/Users/mkonovalov/Documents/VS_code_projects/ITMO/ASCON/kompas_auto_draft/scripts/ksapi_mcp_ssh.sh"]
startup_timeout_sec = 180
tool_timeout_sec = 120
```

Формат конфигурации: [официальная документация OpenAI](https://developers.openai.com/codex/mcp).
Локальный `.mcp.json` содержит такое же подключение для Claude Code и исключён из Git.
Шаблон — [`.mcp.example.json`](../.mcp.example.json); при копировании подставьте
абсолютный путь к SSH-скрипту. После изменения конфигурации
нужно переподключить MCP-клиент, чтобы он заново обнаружил инструменты.

Проверить, что Codex видит настройку, из корня проекта:

```bash
codex mcp get ksapi-expert
```

SSH использует `BatchMode=yes`: ключ должен работать без интерактивного ввода пароля.
При переносе проекта обновите абсолютные пути в конфигурациях и путь общей папки
в `scripts/ksapi_mcp_ssh.sh`. Для другой папки portable MCP есть аргумент `--root`
у `scripts/ksapi_mcp_stdio.py`; также замените путь к встроенному Python в SSH-скрипте.

## Проверка живого КОМПАС

Из терминала Mac:

```bash
ssh -T windows_server '& "C:\MCP\ksapi-expert-mcp\python\python.exe" -I -B -X utf8 "C:\Mac\Home\Documents\VS_code_projects\ITMO\ASCON\kompas_auto_draft\scripts\probe_attach.py"'
```

Ожидаются `GetKompas: attached`, версия, сборка и перечень открытых документов.
Для скриптов используйте указанный Python: системный Python на Windows может иметь
другую архитектуру и не загружать DLL КОМПАС.

Чужие модели и чертежи открываются копией и закрываются без сохранения. Скрипт
`add_center_hole.py` по умолчанию делает пробный прогон; изменение копии требует `--apply`.

## Диагностика

- SSH не подключается: сначала проверьте `ssh -T -o BatchMode=yes windows_server hostname`.
- `Operation not permitted` из shell Codex: возможен запрет сети песочницы. Оператор может
  выполнить read-only диагностику с разрешённым выходом из песочницы; не расширяйте
  разрешения самого построителя и не запускайте его MCP как дочерний процесс shell.
- Не найден скрипт: проверьте доступность общей папки `C:\Mac\Home` и путь проекта.
- Не импортируются зависимости MCP: запускайте встроенным Python, а не системным.
- Не найден КОМПАС: откройте КОМПАС и повторите `probe_attach.py`.
- Медленный первый запуск: MCP загружает локальную embedding-модель и рабочую копию индекса;
  для этого задан увеличенный таймаут инициализации.
