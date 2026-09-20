"""Конвейер: IE → планировщик → инструменты/RAG → ответ → LLM-as-judge."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from rag import BM25Index
from runtime import ApiRun, TARIFF, dump, events, read, usage_cost
from schemas import JudgeVerdict, Plan, RequestExtraction, Resolution
from tools import execute_tool


ROOT = Path(__file__).resolve().parent
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?")


def load_inputs() -> tuple[list[dict], dict[str, dict]]:
    cases = [json.loads(line) for line in (ROOT / "input/requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = {item["id"]: item for item in read(ROOT / "input/gold.json")}
    if len(cases) < 15 or len(cases) != len(gold):
        raise ValueError("Для оценки требуется не менее 15 размеченных обращений без пропусков")
    if {item["id"] for item in cases} != set(gold):
        raise ValueError("Идентификаторы requests.jsonl и gold.json не совпадают")
    return cases, gold


def schema_text(model) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))


def exact_extraction_validator(case: dict):
    def validate(value: RequestExtraction) -> None:
        if value.request_id != case["id"]:
            raise ValueError("request_id не совпадает")
        if value.message_date.isoformat() != case["message_date"]:
            raise ValueError("message_date не совпадает")
        for evidence in value.evidence:
            if evidence.quote not in case["message"]:
                raise ValueError("Цитата evidence отсутствует в исходном обращении")

    return validate


def exact_plan_validator(extraction: RequestExtraction):
    def validate(value: Plan) -> None:
        if value.route != extraction.category:
            raise ValueError("Маршрут плана расходится со структурированным извлечением")
        for call in value.tools:
            if call.name == "search_rules" and not call.arguments.get("query", "").strip():
                raise ValueError("Для поиска нужен query")
            if call.name == "get_contact" and call.arguments.get("category") != value.route:
                raise ValueError("Контакт должен соответствовать маршруту")
            if call.name == "calculate_days":
                if not call.arguments.get("start") or not call.arguments.get("end"):
                    raise ValueError("Для расчёта нужны даты start и end")
                try:
                    start = date.fromisoformat(call.arguments["start"])
                    end = date.fromisoformat(call.arguments["end"])
                except ValueError as error:
                    raise ValueError("Даты инструмента должны быть в формате YYYY-MM-DD") from error
                if end < start:
                    raise ValueError("Конец интервала раньше начала")

    return validate


def citation_corpus(observations: list[dict]) -> dict[str, list[str]]:
    corpus: dict[str, list[str]] = {}
    for observation in observations:
        if observation["name"] != "search_rules":
            continue
        for chunk in observation["result"]:
            corpus.setdefault(chunk["source_id"], []).append(chunk["text"])
    return corpus


def validate_resolution(value: Resolution, case: dict, observations: list[dict]) -> None:
    for quote in value.message_quotes:
        if quote not in case["message"]:
            raise ValueError("Цитата из обращения не является точной")
    corpus = citation_corpus(observations)
    for citation in value.citations:
        if citation.source_id not in corpus:
            raise ValueError("Источник цитаты не был получен поиском")
        if not any(citation.quote in chunk for chunk in corpus[citation.source_id]):
            raise ValueError("Цитата отсутствует в указанном найденном источнике")
    allowed_numbers = case["message"] + "\n" + json.dumps(observations, ensure_ascii=False)
    unsupported = sorted({item for item in NUMBER_RE.findall(value.answer) if item not in allowed_numbers})
    if unsupported:
        raise ValueError("Числа без опоры на обращение или инструмент: " + ", ".join(unsupported))


def append_local(output: Path, event: dict) -> None:
    path = output / "pipeline_trace.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def process_case(run: ApiRun, index: BM25Index, case: dict) -> dict:
    request_block = json.dumps(case, ensure_ascii=False)
    extraction_result = run.call_json(
        f"case/{case['id']}/extract",
        [
            {
                "role": "system",
                "content": (
                    "Ты аналитик обращений магистрантов. Извлеки только сведения из сообщения. "
                    "evidence.quote копируй дословно. requested_deadline — только явно названная будущая дата, "
                    "иначе null. Категории: schedule, documents, tuition, academic_leave, technical. "
                    "Верни только JSON по схеме: " + schema_text(RequestExtraction)
                ),
            },
            {"role": "user", "content": request_block},
        ],
        RequestExtraction,
        post_validate=exact_extraction_validator(case),
        max_tokens=900,
    )
    extraction = RequestExtraction.model_validate(extraction_result["data"])
    append_local(run.output, {"request_id": case["id"], "step": "extract", "attempt": extraction_result["attempt"]})

    plan_result = run.call_json(
        f"case/{case['id']}/plan",
        [
            {
                "role": "system",
                "content": (
                    "Ты планировщик. Разрешены только search_rules, get_contact, calculate_days. "
                    "Всегда сначала вызови search_rules с содержательным запросом и затем get_contact с категорией route. "
                    "calculate_days добавляй, если для ответа полезен интервал между двумя известными датами. "
                    "Не придумывай инструментов. Верни только JSON по схеме: " + schema_text(Plan)
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"request": case, "extraction": extraction.model_dump(mode="json")}, ensure_ascii=False),
            },
        ],
        Plan,
        post_validate=exact_plan_validator(extraction),
        max_tokens=1000,
    )
    plan = Plan.model_validate(plan_result["data"])
    append_local(run.output, {"request_id": case["id"], "step": "plan", "attempt": plan_result["attempt"], "tools": [item.name for item in plan.tools]})

    observations = []
    for call in plan.tools:
        observation = execute_tool(call.name, call.arguments, index)
        observations.append(observation)
        append_local(run.output, {"request_id": case["id"], "step": "tool", "tool": call.name, "arguments": call.arguments})

    resolution_result = run.call_json(
        f"case/{case['id']}/respond",
        [
            {
                "role": "system",
                "content": (
                    "Ты специалист поддержки. Ответь по-русски, спокойно и конкретно. Используй только исходное "
                    "обращение и наблюдения инструментов. Каждая citations.quote должна быть дословным непрерывным "
                    "фрагментом text найденного чанка с тем же source_id. message_quotes копируй дословно. "
                    "Не добавляй чисел, которых нет во входе или наблюдениях; не используй нумерованный список. "
                    "Если данных недостаточно, честно обозначь границу и установи abstained=true. "
                    "Напомни, что база учебная и синтетическая. Верни только JSON по схеме: " + schema_text(Resolution)
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": case,
                        "extraction": extraction.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                        "observations": observations,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        Resolution,
        post_validate=lambda value: validate_resolution(value, case, observations),
        max_tokens=1400,
    )
    resolution = Resolution.model_validate(resolution_result["data"])
    append_local(run.output, {"request_id": case["id"], "step": "respond", "attempt": resolution_result["attempt"], "citations": len(resolution.citations)})

    judge_result = run.call_json(
        f"case/{case['id']}/judge",
        [
            {
                "role": "system",
                "content": (
                    "Ты независимый судья. Оцени: correctness 0–4, groundedness 0–4, path_quality 0–2. "
                    "approved=true только при correctness>=3, groundedness>=3 и path_quality>=1. "
                    "Проверяй соответствие вопросу, найденным правилам и маршруту, но не добавляй новые факты. "
                    "Верни только JSON по схеме: " + schema_text(JudgeVerdict)
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": case,
                        "extraction": extraction.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                        "observations": observations,
                        "resolution": resolution.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        JudgeVerdict,
        max_tokens=800,
    )
    judge = JudgeVerdict.model_validate(judge_result["data"])
    append_local(run.output, {"request_id": case["id"], "step": "judge", "attempt": judge_result["attempt"], "approved": judge.approved})

    return {
        "request": case,
        "extraction": extraction.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "observations": observations,
        "resolution": resolution.model_dump(mode="json"),
        "judge": judge.model_dump(mode="json"),
        "api": {
            "extract": {"response_id": extraction_result["api"]["response_id"], "attempt": extraction_result["attempt"]},
            "plan": {"response_id": plan_result["api"]["response_id"], "attempt": plan_result["attempt"]},
            "respond": {"response_id": resolution_result["api"]["response_id"], "attempt": resolution_result["attempt"]},
            "judge": {"response_id": judge_result["api"]["response_id"], "attempt": judge_result["attempt"]},
        },
    }


def evaluate(results: list[dict], gold: dict[str, dict], output: Path) -> tuple[list[dict], dict]:
    rows = []
    for result in results:
        request_id = result["request"]["id"]
        expected = gold[request_id]
        extraction = result["extraction"]
        plan = result["plan"]
        tools = [item["name"] for item in plan["tools"]]
        retrieved = {
            chunk["source_id"]
            for observation in result["observations"]
            if observation["name"] == "search_rules"
            for chunk in observation["result"]
        }
        checks = {
            "category": extraction["category"] == expected["category"],
            "urgency": expected["urgency"][0] <= extraction["urgency"] <= expected["urgency"][1],
            "sensitive": extraction["contains_sensitive_data"] == expected["sensitive"],
            "route": plan["route"] == expected["category"],
            "source_hit_at_3": bool(retrieved.intersection(expected["expected_sources"])),
            "required_tools": set(expected["required_tools"]).issubset(tools),
            "judge": result["judge"]["approved"],
        }
        rows.append(
            {
                "id": request_id,
                "expected_category": expected["category"],
                "predicted_category": extraction["category"],
                "urgency": extraction["urgency"],
                "tools": tools,
                "retrieved_sources": sorted(retrieved),
                "judge_score": result["judge"]["correctness"] + result["judge"]["groundedness"] + result["judge"]["path_quality"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    trace = [item for item in events(output / "api_trace.jsonl") if item.get("response_id")]
    passed = sum(row["passed"] for row in rows)
    metrics = {
        "cases": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows),
        "category_accuracy": sum(row["checks"]["category"] for row in rows) / len(rows),
        "urgency_accuracy": sum(row["checks"]["urgency"] for row in rows) / len(rows),
        "sensitive_accuracy": sum(row["checks"]["sensitive"] for row in rows) / len(rows),
        "route_accuracy": sum(row["checks"]["route"] for row in rows) / len(rows),
        "source_hit_at_3": sum(row["checks"]["source_hit_at_3"] for row in rows) / len(rows),
        "required_tools_accuracy": sum(row["checks"]["required_tools"] for row in rows) / len(rows),
        "judge_approval_rate": sum(row["checks"]["judge"] for row in rows) / len(rows),
        "mean_judge_score_10": sum(row["judge_score"] for row in rows) / len(rows),
        "api_responses": len(trace),
    }
    return rows, metrics


def controlled_diagnostics(index: BM25Index) -> list[dict]:
    case = {"id": "R00", "message_date": "2026-09-19", "message": "Нужна справка об обучении."}
    good_chunk = index.search(case["message"], top_k=1)[0]
    base = {
        "answer": "Используйте учебный портал.",
        "message_quotes": ["Нужна справка об обучении."],
        "citations": [{"source_id": good_chunk["source_id"], "quote": good_chunk["text"][:40]}],
        "abstained": False,
        "escalation_needed": False,
    }
    diagnostics = []
    scenarios = {
        "ghost_citation": {**base, "citations": [{"source_id": "documents", "quote": "Такой фразы в документе нет"}]},
        "unsupported_number": {**base, "answer": "Ответ гарантирован за 999 дней."},
    }
    for name, payload in scenarios.items():
        blocked = False
        error = None
        try:
            validate_resolution(Resolution.model_validate(payload), case, [{"name": "search_rules", "result": [good_chunk]}])
        except ValueError as exc:
            blocked, error = True, " ".join(str(exc).split())[:300]
        diagnostics.append({"name": name, "blocked": blocked, "error": error})
    blocked = False
    error = None
    try:
        Plan.model_validate(
            {
                "route": "documents",
                "short_rationale": "Проверка запрета фантомного инструмента",
                "tools": [
                    {"name": "search_rules", "arguments": {"query": "справка"}},
                    {"name": "send_email", "arguments": {"to": "student"}},
                    {"name": "get_contact", "arguments": {"category": "documents"}},
                ],
            }
        )
    except ValueError as exc:
        blocked, error = True, " ".join(str(exc).split())[:300]
    diagnostics.append({"name": "phantom_tool", "blocked": blocked, "error": error})
    return diagnostics


def api_totals(output: Path) -> dict:
    trace = [item for item in events(output / "api_trace.jsonl") if item.get("response_id")]
    upper = sum(usage_cost(item.get("usage")) or 0 for item in trace)
    return {
        "api_responses": len(trace),
        "prompt_tokens": sum((item.get("usage") or {}).get("prompt_tokens", 0) for item in trace),
        "completion_tokens": sum((item.get("usage") or {}).get("completion_tokens", 0) for item in trace),
        "responses_without_usage": sum(usage_cost(item.get("usage")) is None for item in trace),
        "estimated_usd_interval": [upper * TARIFF["off_peak_multiplier"], upper],
    }


def write_report(output: Path, results: list[dict], rows: list[dict], metrics: dict, diagnostics: list[dict], run_kind: str) -> None:
    failed = [row for row in rows if not row["passed"]]
    first = results[0]
    lines = [
        "# Academic Navigator: анализ и маршрутизация обращений магистрантов",
        "",
        "**Итоговый проект по дисциплине «Генеративный искусственный интеллект»**",
        "",
        f"Режим результата: **{'реальный API-прогон' if run_kind == 'llm_api' else 'локальный тест с детерминированным транспортом'}**. Дата среза входных данных: 19.09.2026.",
        "",
        "## 1. Постановка задачи",
        "",
        "Цель проекта — построить проверяемый прототип помощника, который принимает свободное обращение студента магистратуры, извлекает структуру запроса, выбирает маршрут, находит опору в локальной базе знаний, формирует ответ и отдельно оценивает его качество. Система не выдаёт себя за официальный сервис университета: все правила и обращения синтетические, а итоговый ответ прямо сообщает об учебном статусе базы.",
        "",
        "Практическая проблема состоит не только в генерации связного текста. Для рабочего сценария необходимо доказать, откуда взят факт, не подменить исходную цитату, не вызвать несуществующий инструмент, заметить чувствительные данные и сохранить путь решения. Поэтому единицей оценки является не один ответ, а полный маршрут от обращения до вердикта судьи.",
        "",
        "## 2. Данные и воспроизводимость",
        "",
        "Набор включает 20 вручную составленных обращений по пяти классам: расписание, документы, оплата, академический отпуск и техническая поддержка. В каждом классе есть обычные и пограничные случаи: неподтверждённая дата из чата, конфликт расписания, публикация паспорта, запрос точной даты возврата, пароль в обращении. Персональные данные реальных людей не использовались.",
        "",
        "База знаний состоит из пяти Markdown-документов. Разметка gold отделена от сообщений и не передаётся модели при решении: она применяется только после формирования ответа. Конфигурация, SHA-256 кода и данных, модель, журнал запросов, usage, кэш и бюджет сохраняются. Повторный запуск с неизменной конфигурацией использует исходные ответы; при изменении кода программа требует новую выходную папку.",
        "",
        "## 3. Архитектура и использованные техники",
        "",
        "В проекте объединены пять изученных подходов:",
        "",
        "- **Structured Information Extraction.** Первая роль возвращает `RequestExtraction`: категорию, намерение, срочность, признак чувствительных данных, дату и точные фрагменты сообщения. Pydantic запрещает лишние поля, ограничивает диапазоны и через `field_validator` отвергает будущий дедлайн, который оказался раньше даты сообщения.",
        "- **RAG.** Инструмент `search_rules` использует прозрачный BM25 по секциям документов и отдаёт три лучших чанка с `source_id`, текстом и оценкой. Ответ не имеет доступа к остальной базе.",
        "- **Агент и инструменты.** Планировщик может выбрать только `search_rules`, `get_contact` и `calculate_days`. Схема требует начинать с поиска и включать проверяемый контакт; фантомный инструмент блокируется до исполнения.",
        "- **Разделение ролей.** Аналитик, планировщик, исполнитель ответа и независимый судья получают разные инструкции и отдельные структурированные схемы. Это упрощает аудит и не позволяет одному неструктурированному сообщению незаметно управлять всем процессом.",
        "- **LLM-as-judge.** Судья выставляет 0–4 за правильность, 0–4 за обоснованность и 0–2 за качество пути. Булево решение связано со шкалами валидатором и не может противоречить выставленным баллам.",
        "",
        "Путь данных: обращение → структурированное извлечение → план → локальные инструменты → ответ с цитатами → независимый вердикт → программная сверка с gold. LLM не исполняет произвольный код, не обращается к интернету и не меняет исходные документы.",
        "",
        "## 4. Защита от галлюцинаций",
        "",
        "После каждой генерации работает не только Pydantic, но и контекстная проверка. Цитата из обращения должна быть точной подстрокой исходного сообщения. Ссылка на правило принимается только тогда, когда указанный `source_id` действительно был получен поиском, а цитата дословно содержится в его чанке. Числа в ответе сопоставляются с обращением и наблюдениями инструментов. Невалидный JSON или нарушенная опора возвращаются модели для исправления не более двух раз; все попытки остаются в журнале.",
        "",
        f"В завершённом наборе проверено {sum(len(item['extraction']['evidence']) for item in results)} цитат из обращений и {sum(len(item['resolution']['citations']) for item in results)} ссылок на правила. Ghost-цитат, пропущенных программной проверкой, — 0. Это утверждение относится к точному совпадению строк, а не доказывает полноту или актуальность синтетической базы.",
        "",
        "## 5. Методика оценки",
        "",
        "Для каждого из 20 примеров измеряются семь условий: категория, допустимый диапазон срочности, признак чувствительных данных, маршрут, попадание ожидаемого источника в top-3, наличие обязательных инструментов и одобрение судьёй. Пример считается пройденным только при выполнении всех семи условий. Отдельно сохраняются траектория инструментов, число API-ответов, токены и расчётный интервал стоимости.",
        "",
        "Эта оценка строже обычной проверки итоговой фразы: удачный текст не компенсирует неверный источник или отсутствующий обязательный шаг. Одновременно LLM-as-judge не считается абсолютной истиной — его решение сопоставляется с детерминированными проверками и ручной разметкой маршрута.",
        "",
        "## 6. Результаты",
        "",
        f"Пройдено **{metrics['passed']} из {metrics['cases']}** примеров; общий pass-rate = **{metrics['pass_rate']:.3f}**. Accuracy категории = {metrics['category_accuracy']:.3f}, маршрута = {metrics['route_accuracy']:.3f}, обнаружения чувствительных данных = {metrics['sensitive_accuracy']:.3f}, hit-rate@3 источника = {metrics['source_hit_at_3']:.3f}, полнота обязательных инструментов = {metrics['required_tools_accuracy']:.3f}. Средний вердикт судьи — {metrics['mean_judge_score_10']:.2f}/10.",
        "",
        "| ID | Ожидалось | Получено | Источник@3 | Инструменты | Judge | Итог |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['expected_category']} | {row['predicted_category']} | "
            f"{'да' if row['checks']['source_hit_at_3'] else 'нет'} | {', '.join(row['tools'])} | "
            f"{row['judge_score']}/10 | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "### Пример сохранённой траектории",
        "",
        f"Для {first['request']['id']} аналитик выбрал категорию `{first['extraction']['category']}` и срочность {first['extraction']['urgency']}. План включил {', '.join(item['name'] for item in first['plan']['tools'])}; поиск вернул {', '.join(chunk['source_id'] for obs in first['observations'] if obs['name'] == 'search_rules' for chunk in obs['result'])}. Проверенных ссылок в ответе: {len(first['resolution']['citations'])}; судья выставил {first['judge']['correctness'] + first['judge']['groundedness'] + first['judge']['path_quality']}/10.",
        "",
        "## 7. Разбор ошибок и контролируемых сбоев",
        "",
    ]
    if failed:
        lines.append("Ниже перечислены реальные непройденные примеры текущего запуска:")
        for row in failed[:3]:
            bad = [name for name, ok in row["checks"].items() if not ok]
            lines.append(f"- **{row['id']}**: не пройдены проверки {', '.join(bad)}. Требуется проверить исходный ответ и запрос в JSON, а не редактировать метрику вручную.")
    else:
        lines.append("Обычный набор не дал непройденных примеров. Поэтому ниже не изображаются вымышленные ошибки модели; отдельно показаны три **контролируемые локальные инъекции**, не входящие в API-eval:")
    for item in diagnostics:
        lines.append(f"- `{item['name']}` — {'заблокировано' if item['blocked'] else 'НЕ заблокировано'}: {item['error'] or 'ошибка отсутствует'}")
    lines += [
        "",
        "Контролируемые проверки показывают действие конкретных ограничителей, но не гарантируют отсутствие всех возможных галлюцинаций. Особенно трудно автоматически проверить корректный пересказ без дословной цитаты, конфликт версий документов и пропуск важного правила, которого не оказалось в top-3.",
        "",
        "## 8. Ресурсы и стоимость",
        "",
        f"Сохранено {metrics['api_responses']} отдельных ответов API, {metrics['prompt_tokens']} входных и {metrics['completion_tokens']} выходных токенов. Расчётный интервал стоимости по зафиксированному тарифному снимку — ${metrics['estimated_usd_interval'][0]:.5f}–${metrics['estimated_usd_interval'][1]:.5f}. Это инженерная оценка по usage, а не сверка с кабинетом провайдера. Ограничители программы: не более 160 вызовов и не более 3 USD оценочного верхнего бюджета.",
        "",
        "При чистом успешном проходе ожидается минимум четыре ответа модели на пример, то есть 80 ответов. Большее значение означает повтор после JSON/schema/grounding-ошибки. Кэш хранит каждый stage отдельно, поэтому один удачный ответ не переиспользуется как независимое повторение другого этапа.",
        "",
        "## 9. Ограничения и переход к production",
        "",
        "Проект является учебным прототипом. База мала и синтетична; BM25 не решает проблему синонимов и версий документов; ручная разметка состоит всего из 20 примеров. Срочность субъективна, а LLM-судья может быть смещён в пользу ответов той же модельной семьи. Контакты зафиксированы в коде и в реальной системе должны поступать из управляемого справочника.",
        "",
        "Для production потребуются: утверждённые источники с датами действия; разграничение доступа; журнал согласий; маскирование персональных данных до LLM; тестирование на prompt injection; человеческая эскалация; мониторинг drift; независимая выборка и второй судья. Финансовые, медицинские и договорные решения нельзя принимать только по ответу модели.",
        "",
        "## 10. Вывод",
        "",
        "Прототип демонстрирует полный проверяемый цикл, а не только генерацию текста. Структурированное извлечение и валидаторы ограничивают форму, RAG даёт локальную опору, инструменты делают маршрут наблюдаемым, разделение ролей снижает смешение обязанностей, а судья и gold-проверки оценивают и ответ, и способ его получения. Главный результат проекта — воспроизводимая траектория с явными границами достоверности.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_manifest(output: Path) -> None:
    excluded = {"artifact_manifest.json", "budget.json"}
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in excluded):
        files[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(output / "artifact_manifest.json", {"algorithm": "sha256", "files": files})


def execute(output="output", transport=None) -> dict:
    output = Path(output)
    run = ApiRun(output, transport=transport)
    cases, gold = load_inputs()
    index = BM25Index(ROOT / "input/knowledge_base")
    results = []
    try:
        for number, case in enumerate(cases, 1):
            print(f"[{number:02d}/{len(cases):02d}] {case['id']}: анализ обращения", flush=True)
            result = process_case(run, index, case)
            dump(output / "cases" / f"{case['id']}.json", result)
            results.append(result)

        rows, metrics = evaluate(results, gold, output)
        diagnostics = controlled_diagnostics(index)
        totals = api_totals(output)
        metrics.update(totals)
        metrics.update(
            {
                "run_kind": run.config["run_kind"],
                "exact_message_quotes": sum(len(item["extraction"]["evidence"]) for item in results),
                "exact_policy_citations": sum(len(item["resolution"]["citations"]) for item in results),
                "ghost_citations": 0,
                "controlled_diagnostics_blocked": sum(item["blocked"] for item in diagnostics),
            }
        )
        dump(output / "results.json", results)
        dump(output / "evaluation.json", rows)
        dump(output / "diagnostics.json", diagnostics)
        dump(output / "metrics.json", metrics)
        write_report(output, results, rows, metrics, diagnostics, run.config["run_kind"])
        run.finish("completed")
        make_manifest(output)
        return {"results": results, "evaluation": rows, "metrics": metrics, "diagnostics": diagnostics}
    except Exception:
        run.finish("interrupted")
        raise
    finally:
        run.close()
