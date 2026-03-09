# 01. Project Brief

## Product goal

Нужно реализовать sales-agent orchestration layer для консультативных B2B-продаж.

Агент должен:
- вести диалог от задачи клиента, а не от линейки моделей;
- выбирать релевантную проблемную ветку разговора;
- доставать доказательства из материалов;
- формировать shortlist только после evidence;
- собирать role-aware follow-up пакет.

## Business problem

Обычный LLM-sales bot почти всегда скатывается в product dump:
- слишком рано предлагает модели;
- выдает цену без понимания конфигурации;
- путает конкурентов, надстройки и TCO;
- не умеет объяснить, почему открыл именно эти документы.

Нужна архитектура, которая делает поведение агента **предсказуемым и проверяемым**.

## Desired behavior

### Agent must:
1. Получить право на диагностику.
2. Собрать минимальные слоты.
3. Зафиксировать branch.
4. Поднять evidence pack.
5. Объяснить evidence.
6. Сделать shortlist.
7. Согласовать следующий шаг.

### Agent must not:
- начинать с продуктовых рекомендаций;
- вызывать tools в opening/discovery;
- слать весь архив документов;
- спорить с конкурентом без comparison evidence.

## First release scope

В первой версии:
- parent graph;
- evidence subgraph;
- mocked / in-memory materials registry adapters;
- 7 eval scenarios;
- traces и acceptance tests.

Не входят:
- CRM write-back;
- email sending;
- quote generation;
- production retrieval infra;
- UI.

## Reference principles

1. **Problem-first** over product-first.
2. **Graph-first orchestration** over free-form agent loops.
3. **Deterministic execution** over unnecessary model autonomy.
4. **Evidence before shortlist**.
5. **Role-aware follow-up**.

## Main branch taxonomy

- `tco`
- `configuration`
- `comparison`
- `service_risk`
- `internal_approval`
- `passenger_route`
- `special_body`
- `special_conditions`
- `unknown_selection`

## Success criteria

The system is successful if:
- it can handle ambiguous inbound requests without premature product recommendation;
- it resolves branch conflicts before evidence retrieval;
- it produces clear, auditable traces;
- it passes the eval suite.

## Suggested implementation order

1. State + enums + transitions.
2. Branch router.
3. Tools adapters.
4. Evidence subgraph.
5. Shortlist + follow-up builders.
6. Tests + eval harness.
