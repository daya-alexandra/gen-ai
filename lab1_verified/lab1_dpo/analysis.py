"""Графики, контроль CSV и выводы, вычисленные по сохранённому прогону."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from schema import Application, CITIES, CUSTOM_ERROR_TYPES, SPECIALITIES, flatten, from_flat


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_distribution(counts: dict, title: str, threshold: float, target: Path, test=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    names, values = list(counts), list(counts.values())
    total = sum(values)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    fig.set_facecolor("#f8fafc")
    bars = ax.barh(names, values, color="#2563eb", height=0.63)
    ax.invert_yaxis()
    ax.axvline(total * threshold, color="#d97706", linestyle="--", linewidth=1.4,
               label=f"Порог {threshold:.0%}")
    ax.set_xlim(0, max(max(values), total * threshold) * 1.24)
    ax.bar_label(bars, labels=[f"{value} ({value / total:.0%})" for value in values], padding=5,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
    ax.set_xlabel("Количество заявок")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(("ТЕСТОВЫЕ ДАННЫЕ · " if test else "Синтетические заявки ДПО · ") + title,
                 loc="left", fontweight="bold", pad=16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_conclusions(output: Path, metrics: dict, baseline: dict | None = None):
    test = metrics["run_kind"] != "llm_api"
    prefix = "ТЕСТОВЫЙ ПРОГОН БЕЗ LLM: эти результаты нельзя сдавать как эксперимент.\n\n" if test else ""
    mode = metrics["strategy"]
    intro = ("Применена стратификация: по 5 заявок для каждого из 10 городов; порядок городов перемешан."
             if mode == "stratified" else "Город выбирался случайно из списка 10 городов.")
    paragraph1 = (
        f"Получено и повторно проверено {metrics['count']}/50 синтетических заявок. {intro} "
        f"Максимальная доля города — {metrics['max_city_share']:.0%} (порог 40%), "
        f"специальности — {metrics['max_speciality_share']:.0%} (порог 35%). "
        f"Представлено {metrics['specialities_present']} из {len(SPECIALITIES)} специальностей; "
        f"повторяющихся ФИО сверх первого вхождения — {metrics['duplicate_names']}. "
    )
    if baseline:
        paragraph1 += (
            f"При случайном выборе города максимальные доли были "
            f"{baseline['max_city_share']:.0%} по городам и {baseline['max_speciality_share']:.0%} "
            f"по специальностям; со стратификацией — {metrics['max_city_share']:.0%} и "
            f"{metrics['max_speciality_share']:.0%}. Это сравнение двух небольших прогонов: "
            "различия специальностей нельзя уверенно приписать только квотированию. "
        )
    paragraph1 += (
        "Городские квоты управляют географией, но сами по себе не обеспечивают баланс специальностей. "
        "Выборка отражает поведение одной модели; она не репрезентативна для реальных заявителей. "
        "Географическая принадлежность районов и соответствие профессии курсу автоматически не подтверждены."
    )
    caught = metrics["custom_validator_error_events"]
    if caught:
        observed = (f"Пользовательские @field_validator выявили ошибки в {caught} ответах; "
                    f"по типам: {metrics['custom_error_types']}. "
                    "В журнале сохранены исходные ответы и причины отклонения. ")
    else:
        observed = (
            "Пользовательские @field_validator не обнаружили нарушений в ответах этого прогона. "
            "Это наблюдение по журналу всех попыток, а не предположение по итоговому CSV. "
        )
    paragraph2 = (
        observed + f"Всего ошибок JSON/Pydantic — {metrics['validation_error_events']}; "
        f"несовпадений с назначенным городом — {metrics['seed_mismatches']}. "
        "Для исправления отклонённых ответов предусмотрен повторный запрос через "
        "response_model=Application и max_retries=3; ручная правка полей не используется. "
        "Работоспособность проверок отдельно проверена на намеренно некорректных тестовых объектах; "
        "эти тесты не включены в статистику ошибок LLM. "
        f"Пороговые критерии задания {'соблюдены' if metrics['meets_good_criteria'] else 'НЕ соблюдены полностью'}; "
        "подробные числа и ограничения зафиксированы в metrics.json и trace.jsonl."
    )
    (output / "выводы.md").write_text(prefix + paragraph1 + "\n\n" + paragraph2 + "\n", encoding="utf-8")


def analyze_run(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    records = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(records) != 50:
        raise ValueError("Для итоговых артефактов нужен завершённый прогон из 50 заявок")
    applications = [Application.model_validate(row) for row in records]
    rows = [flatten(app) for app in applications]
    with (output / "applications.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "applications.csv").open(encoding="utf-8-sig", newline="") as handle:
        restored = [from_flat(row) for row in csv.DictReader(handle)]
    if [app.model_dump() for app in restored] != records:
        raise ValueError("CSV после чтения не совпал с исходными заявками")
    trace = [json.loads(line) for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    attempts = [event for event in trace if event["event"] == "llm_attempt"]
    errors = [event for event in attempts if event["status"] == "validation_error"]
    custom = [event for event in errors if any(error["type"] in CUSTOM_ERROR_TYPES
                                              for error in event.get("errors", []))]
    city_counts = Counter(app.address.city for app in applications)
    speciality_counts = Counter(app.speciality for app in applications)
    name_counts = Counter(app.full_name for app in applications)
    max_city = max(city_counts.values()) / len(applications)
    max_speciality = max(speciality_counts.values()) / len(applications)
    successful_api = [event for event in attempts if event["status"] != "api_error"]
    missing_usage = sum(event.get("prompt_tokens") is None or event.get("completion_tokens") is None
                        for event in successful_api)
    metrics = dict(
        run_kind=manifest["run_kind"], model=manifest["config"]["model"],
        strategy=manifest["config"]["strategy"], count=len(applications),
        city_counts={city: city_counts[city] for city in CITIES},
        speciality_counts={speciality: speciality_counts[speciality] for speciality in SPECIALITIES},
        max_city_share=max_city, max_speciality_share=max_speciality,
        specialities_present=len(speciality_counts),
        duplicate_names=sum(count - 1 for count in name_counts.values()),
        validation_error_events=len(errors), custom_validator_error_events=len(custom),
        custom_error_types=dict(Counter(error["type"] for event in custom
                                       for error in event["errors"] if error["type"] in CUSTOM_ERROR_TYPES)),
        seed_mismatches=sum(event["event"] == "seed_mismatch" for event in trace),
        api_calls=len(attempts), api_errors=sum(event["status"] == "api_error" for event in attempts),
        prompt_tokens_observed=sum(event.get("prompt_tokens") or 0 for event in attempts),
        completion_tokens_observed=sum(event.get("completion_tokens") or 0 for event in attempts),
        responses_without_usage=missing_usage,
        execution_seconds=manifest.get("execution_seconds"),
        meets_good_criteria=len(applications) == 50 and max_city <= 0.4 and max_speciality <= 0.35,
    )
    if metrics["strategy"] == "stratified" and any(city_counts[city] != 5 for city in CITIES):
        raise ValueError("Нарушена стратификация: должно быть ровно по 5 заявок на город")
    save_json(output / "metrics.json", metrics)
    test = manifest["run_kind"] != "llm_api"
    plot_distribution(metrics["city_counts"], "города", 0.4, output / "cities.png", test)
    plot_distribution(metrics["speciality_counts"], "специальности", 0.35,
                      output / "specialities.png", test)
    write_conclusions(output, metrics)
    return metrics


def compare_runs(baseline_dir: Path, stratified_dir: Path) -> dict:
    a, b = analyze_run(baseline_dir), analyze_run(stratified_dir)
    meta_a = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    meta_b = json.loads((stratified_dir / "run.json").read_text(encoding="utf-8"))
    if a["strategy"] != "random" or b["strategy"] != "stratified":
        raise ValueError("Нужны стратегии random и stratified")
    for key in ("model", "temperature", "seed", "code", "count"):
        if meta_a["config"][key] != meta_b["config"][key]:
            raise ValueError(f"Несопоставимые настройки: {key}")
    if a["run_kind"] != b["run_kind"]:
        raise ValueError("Нельзя сравнивать тестовые объекты с LLM-экспериментом")
    write_conclusions(stratified_dir, b, baseline=a)
    lines = ["# Сравнение случайного выбора и стратификации", ""]
    if a["run_kind"] != "llm_api":
        lines += ["ТЕСТОВЫЕ ДАННЫЕ, БЕЗ ВЫЗОВОВ МОДЕЛИ. НЕ ДЛЯ СДАЧИ.", ""]
    lines += [
        "Два прогона по 50 синтетических заявок. Модель, шаблон промпта, температура и seed одинаковы.",
        "Seed воспроизводит расписание городов и порядок вариантов; ответы API могут различаться.", "",
        "| Показатель | Случайный город | Квоты по городам |", "|---|---:|---:|",
        f"| Максимальная доля города | {a['max_city_share']:.0%} | {b['max_city_share']:.0%} |",
        f"| Максимальная доля специальности | {a['max_speciality_share']:.0%} | {b['max_speciality_share']:.0%} |",
        f"| Представлено специальностей | {a['specialities_present']} | {b['specialities_present']} |",
        f"| Ответов с ошибками пользовательских валидаторов | {a['custom_validator_error_events']} | {b['custom_validator_error_events']} |",
        "", "| Специальность | Случайный город | Квоты по городам |", "|---|---:|---:|",
    ]
    for name in SPECIALITIES:
        lines.append(f"| {name} | {a['speciality_counts'][name]} | {b['speciality_counts'][name]} |")
    lines += ["", "По каждому режиму выполнен один прогон. Наблюдаемые различия не доказывают причинный эффект "
              "стратификации на специальности. Прямой эффект квотирования — ровно 5 заявок для каждого города.",
              "Для вывода о стабильном влиянии на профессии нужны повторные независимые прогоны."]
    (stratified_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = (compare_runs(args.baseline, args.output) if args.baseline else analyze_run(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
