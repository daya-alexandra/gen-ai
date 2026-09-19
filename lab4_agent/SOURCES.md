# Источники и границы эксперимента

Код подготовлен по материалам [paNikitin/gen-ai](https://github.com/paNikitin/gen-ai) и их копии в [daya-alexandra/gen-ai](https://github.com/daya-alexandra/gen-ai). Требования сохранены в исходных папках семинаров; они не изменяются этой работой.

- [Домашнее задание семинара 5](https://github.com/daya-alexandra/gen-ai/blob/main/семинар_5/домашнее_задание.md), blob `929fcbcda745ecfb8879c123163674981de72d44`.
- [Домашнее задание семинара 6](https://github.com/daya-alexandra/gen-ai/blob/main/семинар_6/домашнее_задание.md), blob `92e50f355e65182167c9e39bd194910a43085486`.
- Стартеры семинаров 5–6: имена инструментов, исходные вопросы, структуры Plan/SubQuestion/WorkerAnswer/Verdict и общий алгоритм ReAct/PWC. Код выполнения, бюджет, проверка и упаковка доработаны для воспроизводимого запуска. Сырые результаты примера преподавателя `eval_pwc_results.json` не используются как собственные.
- JSON-клиент с кэшем и аудитом адаптирован из нашей третьей лабораторной; native tool calls добавлены по [документации DeepSeek](https://api-docs.deepseek.com/guides/tool_calls/). Планировщик/критик используют [JSON Output](https://api-docs.deepseek.com/guides/json_mode/).
- [Тарифы DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) проверены 19.09.2026. Используется `deepseek-flash` с явно отключённым thinking. Цены служат оценке; списания в аккаунте отдельно не проверяются.

Четыре CSV из `семинар_5/starter/data` скопированы побайтно:

| Файл | Git blob SHA-1 |
|---|---|
| fx_benchmark.csv | `afdd6f92486376b68f664ccf1e5879a27a0b2556` |
| key_rate_history.csv | `ca03fcc66991f98eaa04808dbfd4bf06160caa11` |
| cpi_ru_monthly.csv | `c32bd060841830cdbdd2af86f0e091501ee00fc3` |
| unemployment_ru_monthly.csv | `05a71499fffceeca7c69c40c83ff1b35df4dbb95` |

CSV — учебные фикстуры, не независимая выгрузка официальной статистики. `key_rate_history.csv` содержит `mock_cut`; в инфляции `mock_from_here` начинается с 2025-05. Для остальных строк первоисточник здесь заново не удостоверялся. Учебный режим явно раскрывает fixture, synthetic и фактическую дату записи.

Содержимое источников не изменяется, но доступ к FX скорректирован относительно стартера: выбирается запись не позже запроса, а не ближайшая по абсолютному расстоянию. Калькулятор использует ограниченный AST вместо eval/sympify. Схемы и реестр инструментов проверяются до вызова функции. Итоговые данные API появятся только после пользовательского запуска.

