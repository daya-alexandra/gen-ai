"""Проверка ключа и списка моделей без генерации заявок."""

from llm_client import explain_error, get_model, make_client


def main():
    client = None
    try:
        client = make_client()
        models = [model.id for model in client.raw_client.models.list().data]
        print("Ключ принят. Доступные модели: " + ", ".join(models))
        model = get_model()
        if model not in models:
            print(f"Модели {model} нет в списке. Уточни LLM_MODEL перед генерацией.")
            return 2
        print(f"Модель {model} доступна. Список моделей не подтверждает достаточность баланса.")
        return 0
    except Exception as exc:
        print(explain_error(exc))
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
