"""Все эксперименты первого семинара с сохранением воспроизводимых артефактов."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from runtime import ApiRun, TARIFF, dump, events, usage_cost

ROOT = Path(__file__).resolve().parent
OWN_TEXT = (
    "В магистерском исследовании рассматривается применение генеративного искусственного "
    "интеллекта для анализа обращений студентов. Особое внимание уделяется проверке "
    "источников, воспроизводимости эксперимента и ограничениям автоматической оценки."
)
TEMPERATURE_PROMPT = "Назови одним словом главную проблему цифрового образования. Ответь одним словом."
CLICKBAIT = "ВЫ НЕ ПОВЕРИТЕ, что произошло с рублём за одну ночь!"
TEST_HEADLINES = [
    "Центробанк повысил ключевую ставку до 21% годовых",
    CLICKBAIT,
    "Минфин опубликовал проект федерального бюджета на 2027 год",
    "10 акций, которые сделают вас миллионером в 2027 году",
    "Уровень безработицы снизился до 2,3%",
]
PERSONAS = {
    "professor": "Ты профессор финансов. Объясняй академично, называй риски и не давай персональных инвестиционных советов.",
    "friend": "Ты друг-аспирант. Обращайся на ты, пиши разговорно и не больше пяти коротких предложений.",
    "consultant": "Ты корпоративный wealth advisor. Дай структурированный ответ, используй 2–3 английских термина и закончи дисклеймером о рисках.",
    "child": "Объясни ребёнку десяти лет простыми словами и жизненной аналогией, без терминов, не больше пяти предложений.",
}
DIGIT = re.compile(r"^\s*(10|[1-9])\s*$")


def token_estimate(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def score_from_text(text: str) -> int:
    match = re.search(r"(?<!\d)(10|[1-9])(?!\d)", text)
    return int(match.group(1)) if match else -1


def load_headlines() -> list[dict]:
    with (ROOT / "input/headlines.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def api_totals(output: Path) -> dict:
    log = [item for item in events(output / "api_trace.jsonl") if item.get("response_id")]
    upper = sum(usage_cost(item.get("usage")) or 0 for item in log)
    return {
        "api_responses": len(log),
        "prompt_tokens": sum((item.get("usage") or {}).get("prompt_tokens", 0) for item in log),
        "completion_tokens": sum((item.get("usage") or {}).get("completion_tokens", 0) for item in log),
        "missing_usage": sum(usage_cost(item.get("usage")) is None for item in log),
        "estimated_usd_interval": [upper * TARIFF["off_peak_multiplier"], upper],
    }


def make_plots(output: Path, temperatures: dict, scored: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["T=0", "T=1.0"]
    pct = [temperatures["t0_unique_share"] * 100, temperatures["t1_unique_share"] * 100]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, pct, color=["#4A90D9", "#F96167"], width=0.55)
    ax.set_ylabel("Уникальные ответы, %")
    ax.set_ylim(0, 110)
    ax.set_title("Влияние temperature")
    for bar, value in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output / "temperature_experiment.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    means = [row["score_mean"] for row in scored]
    stds = [row["score_std"] for row in scored]
    colors = ["#4A90D9" if value <= 4 else "#F9A825" if value <= 7 else "#F96167" for value in means]
    positions = list(range(len(scored)))
    ax.barh(positions, means, xerr=stds, color=colors, error_kw={"capsize": 2})
    ax.set_yticks(positions)
    ax.set_yticklabels([row["headline"][:55] for row in scored], fontsize=7)
    ax.set_xlim(0, 11)
    ax.set_xlabel("Кликбейтность: среднее ± стандартное отклонение")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "headline_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, result: dict) -> None:
    temp = result["temperature"]
    golf = result["prompt_experiments"]["golf"]
    scored = result["headline_scores"]
    unstable = sorted(scored, key=lambda row: row["score_std"], reverse=True)[:3]
    lines = [
        "# Лабораторная № 1. Основы API и промптинга",
        "",
        "**Режим: реальный API-прогон**" if result["run_kind"] == "llm_api" else "**Режим: локальный тест без API**",
        "",
        "## 1. API и токены",
        "",
        f"Первый запрос выполнен моделью `{result['hello']['server_model']}`. Ответ: «{result['hello']['text']}».",
        f"Собственный текст содержит {result['token_estimate']['characters']} символов, {result['token_estimate']['words']} слов и примерно {result['token_estimate']['tokens']} токенов cl100k_base.",
        "",
        "## 2. Temperature",
        "",
        f"При T=0 получено {temp['t0_unique']} уникальных ответов из 10 ({temp['t0_unique_share']:.0%}); при T=1,0 — {temp['t1_unique']} из 10 ({temp['t1_unique_share']:.0%}). Это один небольшой эксперимент: он показывает поведение текущей модели, но не доказывает общую детерминированность или креативность.",
        "",
        "## 3. Промптинг",
        "",
        f"В prompt golf чистая цифра получена для {golf['clean']}/5 заголовков. Лестница A сохраняет ответы zero-shot, role, format и few-shot; битва персон содержит четыре независимых ответа на один вопрос.",
        "",
        "## 4. Оценка заголовков",
        "",
        f"Оценены {len(scored)} заголовков, каждый по 10 раз. Формат не удалось разобрать в {sum(row['invalid_scores'] for row in scored)} ответах.",
        "",
        "Самые нестабильные заголовки:",
    ]
    for row in unstable:
        lines.append(f"- {row['id']}: {row['score_mean']:.2f} ± {row['score_std']:.2f} — {row['headline']}")
    totals = result["totals"]
    lines += [
        "",
        "## 5. Вывод",
        "",
        "System-промпт заметно влияет и на формат, и на тон ответа. Даже temperature=0 не следует считать математической гарантией одинакового результата: провайдер и реализация декодирования могут меняться. Для числового production-вывода нужна структурированная схема, а не регулярное выражение — это ограничение явно видно в оценщике заголовков.",
        "",
        f"Получено {totals['api_responses']} ответов API, {totals['prompt_tokens']} входных и {totals['completion_tokens']} выходных токенов. Расчётная стоимость: ${totals['estimated_usd_interval'][0]:.5f}–${totals['estimated_usd_interval'][1]:.5f}; серверный биллинг отдельно не проверялся.",
        "",
        "Полные ответы находятся в JSON, исходные события — в api_trace.jsonl, графики — в PNG. Данные не редактировались вручную после API-прогона.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(output="output", transport=None) -> dict:
    output = Path(output)
    run = ApiRun(output, transport=transport)
    try:
        hello = run.call(
            "hello",
            [
                {"role": "system", "content": "Ты помощник исследователя экономики."},
                {"role": "user", "content": "Объясни в одном предложении, что такое инфляция."},
            ],
            temperature=0,
        )
        own_tokens = token_estimate(OWN_TEXT)
        prices = {name: round(100 * (own_tokens * inp + own_tokens * out) / 1_000_000, 6) for name, (inp, out) in {
            "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "deepseek-flash": (0.30, 1.20)
        }.items()}
        token_result = {"text": OWN_TEXT, "characters": len(OWN_TEXT), "words": len(OWN_TEXT.split()), "tokens": own_tokens, "cost_100_full_responses_usd": prices}

        t0 = [run.call(f"temperature/t0/{i:02d}", [{"role": "user", "content": TEMPERATURE_PROMPT}], 0, 50) for i in range(1, 11)]
        t1 = [run.call(f"temperature/t1/{i:02d}", [{"role": "user", "content": TEMPERATURE_PROMPT}], 1.0, 50) for i in range(1, 11)]
        t0_set = {item["text"].casefold() for item in t0}
        t1_set = {item["text"].casefold() for item in t1}
        temperature = {
            "prompt": TEMPERATURE_PROMPT,
            "t0": t0,
            "t1": t1,
            "t0_unique": len(t0_set),
            "t1_unique": len(t1_set),
            "t0_unique_share": len(t0_set) / 10,
            "t1_unique_share": len(t1_set) / 10,
        }

        task = f"Оцени кликбейтность заголовка от 1 до 10: «{CLICKBAIT}»"
        role = "Ты строгий главный редактор делового издания и специалист по медиаграмотности."
        format_prompt = role + " Ответь строго: число — одно короткое предложение причины. Без Markdown и преамбулы."
        fewshot = [
            {"role": "system", "content": format_prompt},
            {"role": "user", "content": "Оцени кликбейтность: «Росстат опубликовал данные об инфляции»"},
            {"role": "assistant", "content": "1 — нейтральный информационный заголовок без манипуляции."},
            {"role": "user", "content": "Оцени кликбейтность: «ШОК! Этот секрет изменит вашу жизнь навсегда»"},
            {"role": "assistant", "content": "10 — капс, недосказанность и необоснованное обещание."},
            {"role": "user", "content": task},
        ]
        ladder = {
            "zero_shot": run.call("prompt/ladder/zero", [{"role": "user", "content": task}], 0.3, 220),
            "role": run.call("prompt/ladder/role", [{"role": "system", "content": role}, {"role": "user", "content": task}], 0.3, 220),
            "format": run.call("prompt/ladder/format", [{"role": "system", "content": format_prompt}, {"role": "user", "content": task}], 0.3, 220),
            "fewshot": run.call("prompt/ladder/fewshot", fewshot, 0.3, 220),
        }
        golf_prompt = "Кликбейт 1–10. Только число."
        golf_items = []
        for index, headline in enumerate(TEST_HEADLINES, 1):
            answer = run.call(f"prompt/golf/{index:02d}", [{"role": "system", "content": golf_prompt}, {"role": "user", "content": headline}], 0, 20)
            golf_items.append({"headline": headline, "answer": answer, "clean_digit": bool(DIGIT.fullmatch(answer["text"]))})
        golf = {"system": golf_prompt, "items": golf_items, "clean": sum(item["clean_digit"] for item in golf_items)}

        persona_question = "Стоит ли студенту с 50 000 рублей покупать акции российских банков?"
        persona_results = {
            name: run.call(f"prompt/persona/{name}", [{"role": "system", "content": system}, {"role": "user", "content": persona_question}], 0.7, 350)
            for name, system in PERSONAS.items()
        }

        score_prompt = (
            "Ты эксперт по медиа. Оцени кликбейтность заголовка от 1 до 10: "
            "1 — нейтральный, 10 — максимальный кликбейт. Ответь только числом."
        )
        scored = []
        for row in load_headlines():
            raw_scores = []
            raw_answers = []
            for repeat in range(1, 11):
                answer = run.call(f"scorer/{row['id']}/{repeat:02d}", [{"role": "system", "content": score_prompt}, {"role": "user", "content": row["headline"]}], 0, 20)
                raw_answers.append(answer)
                raw_scores.append(score_from_text(answer["text"]))
            valid = [value for value in raw_scores if 1 <= value <= 10]
            mean = sum(valid) / len(valid) if valid else math.nan
            std = math.sqrt(sum((value - mean) ** 2 for value in valid) / len(valid)) if valid else math.nan
            scored.append({**row, "scores_raw": raw_scores, "score_mean": mean, "score_std": std, "invalid_scores": 10 - len(valid), "answers": raw_answers})

        with (output / "headlines_scored.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "headline", "score_mean", "score_std", "invalid_scores", "scores_raw"])
            writer.writeheader()
            for row in scored:
                writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if key == "scores_raw" else row[key] for key in writer.fieldnames})

        prompt_experiments = {"ladder": ladder, "golf": golf, "personas": persona_results}
        dump(output / "hello.json", hello)
        dump(output / "token_estimate.json", token_result)
        dump(output / "temperature.json", temperature)
        dump(output / "prompt_experiments.json", prompt_experiments)
        dump(output / "headline_scores.json", scored)
        make_plots(output, temperature, scored)
        totals = api_totals(output)
        result = {
            "run_kind": run.config["run_kind"],
            "hello": hello,
            "token_estimate": token_result,
            "temperature": temperature,
            "prompt_experiments": prompt_experiments,
            "headline_scores": scored,
            "totals": totals,
        }
        metrics = {
            "run_kind": run.config["run_kind"],
            "temperature_unique": {"t0": len(t0_set), "t1": len(t1_set)},
            "prompt_golf_clean": golf["clean"],
            "headline_count": len(scored),
            "headline_repeats": 10,
            "invalid_scores": sum(row["invalid_scores"] for row in scored),
            **totals,
        }
        dump(output / "metrics.json", metrics)
        write_report(output, result)
        run.finish("completed")
        return result
    except Exception:
        run.finish("interrupted")
        raise
    finally:
        run.close()
