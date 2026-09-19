# Практическое применение генеративного ИИ

Форк [репозитория курса](https://github.com/paNikitin/gen-ai) с выполненными лабораторными работами.

## Лабораторная № 1 — синтетические заявки на курсы ДПО

[Код, результаты и инструкция](lab1_verified/README.md).

Генерация выполнена 19.09.2026 через DeepSeek API. Основной набор содержит 50 валидных заявок: по 5 на каждый из 10 городов. Максимальная доля специальности — 32% при пороге 35%. Для сравнения сохранён отдельный случайный прогон из 50 заявок.

- [Итоговые заявки CSV](lab1_verified/lab1_dpo/output/stratified/applications.csv)
- [Выводы](lab1_verified/lab1_dpo/output/stratified/выводы.md)
- [Сравнение стратегий](lab1_verified/lab1_dpo/output/stratified/comparison.md)
- [Проверка результатов](lab1_verified/VERIFICATION.md)

![Распределение по городам](lab1_verified/lab1_dpo/output/stratified/cities.png)

![Распределение по специальностям](lab1_verified/lab1_dpo/output/stratified/specialities.png)

## Материалы курса

Исходные материалы преподавателя находятся в папках `семинар_1`–`семинар_6` и `финальный_проект`.
