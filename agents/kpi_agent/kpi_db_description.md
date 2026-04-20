# База данных `kpi.sqlite`

Файл базы данных: `C:\Projects\GWP\kpi\kpi.sqlite`

База описывает систему KPI филиала: штатную структуру, назначения KPI на организационно-ролевые контексты и методики расчета KPI.

Основные таблицы:

- `kpi_values` - таблица фактов назначений KPI на штатную структуру, центры ответственности и уровни должностей.
- `kpi_staff_structure` - нормализованная структура подразделений, групп работников и должностей. Содержит уникальные комбинации `department_1` ... `department_8`, `employee_group` и `position`.
- `kpi_method` - справочник методик расчета KPI.

Текущий размер базы:

- `kpi_method`: 30 строк.
- `kpi_staff_structure`: 129 строк.
- `kpi_values`: 986 строк.

## Общие правила интерпретации данных

1. Основная таблица для анализа KPI - `kpi_values`.
   Она хранит связи между KPI, центрами ответственности, уровнями должностей, детализацией расчета и нормализованной штатной структурой.

2. Подразделения, группа работников и должность не хранятся напрямую в `kpi_values`.
   Для получения этих полей нужно присоединять `kpi_staff_structure` по `kpi_values.staff_structure_ref = kpi_staff_structure.staff_structure_id`.

3. Методика расчета не должна присоединяться по названию KPI.
   Используй `kpi_values.kpi_method_ref = kpi_method.source_row`, потому что название KPI может отличаться пробелами, сносками и уточнениями.

4. `kpi_method_ref` может быть пустым.
   В текущих данных 6 строк `kpi_values` не имеют ссылки на методику. Это строки с KPI `Карта КПЭ`.

5. `source_row` - это технический идентификатор записи, а не бизнес-идентификатор.
   Его можно использовать для устойчивой сортировки и трассировки записи внутри базы.

6. Пустые значения обычно хранятся как `NULL`.
   Исключение: если не задана `position`, в `kpi_staff_structure.position` записывается `сотрудник`.
   Для уровней подразделений `NULL` остается нормальной особенностью структуры.

7. Таблица `kpi_staff_structure` содержит уникальные комбинации подразделений, группы работников и должности.
   Уникальность задается по всем полям `department_1` ... `department_8`, `employee_group` и `position`; при проверке уникальности `NULL` трактуется как пустая строка.

8. `responsibility_center` и `position_group` остаются в `kpi_values`.
   Эти поля описывают контекст назначения KPI, а не саму структурную единицу штатного плана.
   `position_group` означает уровень должности для назначения KPI, например исполнитель, руководитель среднего звена или руководитель верхнего звена. Это не то же самое, что `kpi_staff_structure.employee_group`, где хранится группа работников внутри подразделения.

9. KPI определяется сочетанием подразделения, группы работников и должности.
   В базе это сочетание хранится в `kpi_staff_structure`.

10. Поля `analytics_1`, `analytics_2`, `analytics_3`, `analytics_4` есть в схеме `kpi_values`, но в текущей базе все они пустые.
   Их не стоит использовать как обязательный источник данных, пока в базе нет заполненных значений.

11. Для построения штатного плана начинай с `kpi_staff_structure`.
    Для анализа KPI по штатной структуре присоединяй к ней `kpi_values`.

## Связи между таблицами

```text
kpi_staff_structure.staff_structure_id
    -> kpi_values.staff_structure_ref

kpi_method.source_row
    -> kpi_values.kpi_method_ref
```

Фактически `kpi_values` является центральной таблицей фактов:

```text
kpi_staff_structure 1 ---- * kpi_values * ---- 0..1 kpi_method
```

`kpi_values.staff_structure_ref` всегда заполнен в текущей версии данных.
`kpi_values.kpi_method_ref` может быть `NULL`.

## Таблица `kpi_staff_structure`

Таблица содержит нормализованную штатную структуру: уникальные цепочки подразделений, группу работников и должность.
Одна строка описывает один структурный путь от верхнего уровня подразделения до группы работников и должности.
Поля `department_1` ... `department_8` отражают иерархию подразделений в организационной структуре:
`department_1` - самый верхний заполненный уровень, далее уровни детализируются вниз по структуре.

Полное наименование позиции всегда включает `position`, `employee_group` и все заполненные поля `department_*`.
Для человекочитаемого вывода его нужно собирать от должности к ближайшему подразделению и дальше вверх по иерархии:
`position`, затем `employee_group` в скобках, затем заполненные `department_8` ... `department_1`.
Например:

- Для позиции начальника отдела информационных технологий: `Начальник отдела (Сопровождение функциональное (РС)) Отдел информационных технологий Филиала СПАО "Ингосстрах" в г. Санкт-Петербурге`.
- Для назначения KPI с `source_row = 182`: `исполнитель: сотрудник (Руководство центров продаж (дополнительный офис)) Операционный офис в г. Луга Отдел по работе с операционными офисами Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге`.
  Префикс `исполнитель:` берется из `kpi_values.position_group`, а остальная часть имени - из `kpi_staff_structure`.

### Поля

- `staff_structure_id`
  Внутренний идентификатор строки штатной структуры.
  Используется как внешний ключ из `kpi_values.staff_structure_ref`.

- `department_1`
  Первый уровень подразделения.
  Это верхний уровень иерархии подразделений.
  В текущей базе это верхний уровень филиала: `Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге`.

- `department_2`
  Второй уровень подразделения.
  Обычно отдел или крупное подразделение внутри филиала.
  Подчинен `department_1`.
  Может быть `NULL`, если второй уровень не задан.

- `department_3`
  Третий уровень подразделения.
  Подчинен `department_2`.
  Может быть `NULL`.

- `department_4`
  Четвертый уровень подразделения.
  Подчинен `department_3`.
  В текущих данных заполнен редко.

- `department_5`
  Пятый уровень подразделения.
  Подчинен `department_4`.
  В текущих данных полностью пустой, но поле оставлено для совместимости со структурой данных.

- `department_6`
  Шестой уровень подразделения.
  Подчинен `department_5`.
  В текущих данных полностью пустой.

- `department_7`
  Седьмой уровень подразделения.
  Подчинен `department_6`.
  В текущих данных полностью пустой.

- `department_8`
  Восьмой уровень подразделения.
  Подчинен `department_7`.
  В текущих данных полностью пустой.

- `employee_group`
  Группа работников.
  Это часть штатной структуры и один из признаков, по которым определяется применимость KPI.
  Используется вместе с иерархией подразделений и `position`.

- `position`
  Должность.
  Если должность не задана, используется значение `сотрудник`.
  Поэтому в текущей базе `position` не содержит `NULL`.

### Как использовать таблицу `kpi_staff_structure`

- Используй эту таблицу как основу для штатного плана.
- Для вывода полной структуры сортируй по `staff_structure_id`: идентификаторы задают стабильный порядок строк штатной структуры.
- Если нужен список подразделений без должностей, группируй по `department_1` ... `department_8`.
- Если нужен список должностей по подразделению, фильтруй по нужным `department_*` и смотри поля `employee_group` и `position`.
- Для полного наименования позиции выводи `position`, `employee_group` и все заполненные уровни `department_*`.
  Практический порядок для чтения: `position`, затем `employee_group` в скобках, затем `department_8` ... `department_1` с пропуском пустых уровней.
- Чтобы понять, какие KPI относятся к структурной строке, присоединяй `kpi_values` по `staff_structure_ref`.

Пример запроса для штатной структуры с количеством строк KPI:

```sql
SELECT
    s.staff_structure_id,
    s.department_1,
    s.department_2,
    s.department_3,
    s.employee_group,
    s.position,
    COUNT(v.source_row) AS kpi_row_count
FROM kpi_staff_structure AS s
LEFT JOIN kpi_values AS v
    ON v.staff_structure_ref = s.staff_structure_id
GROUP BY
    s.staff_structure_id,
    s.department_1,
    s.department_2,
    s.department_3,
    s.employee_group,
    s.position
ORDER BY s.staff_structure_id;
```

Пример запроса для получения полной структуры подразделений внутри филиала
`Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге`:

```sql
SELECT DISTINCT
    department_1,
    department_2,
    department_3,
    department_4,
    department_5,
    department_6,
    department_7,
    department_8
FROM kpi_staff_structure
WHERE department_1 = 'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге'
ORDER BY
    department_1,
    department_2,
    department_3,
    department_4,
    department_5,
    department_6,
    department_7,
    department_8;
```

Пример запроса для получения всех позиций, работающих непосредственно в управлении
`Управление добровольного медицинского страхования Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге`.
В этом запросе выбираются только строки самого управления, без дочерних подразделений:

```sql
SELECT
    s.staff_structure_id,
    s.employee_group,
    s.position,
    TRIM(
        s.position || ' ' ||
        '(' || s.employee_group || ') ' ||
        COALESCE(s.department_8 || ' ', '') ||
        COALESCE(s.department_7 || ' ', '') ||
        COALESCE(s.department_6 || ' ', '') ||
        COALESCE(s.department_5 || ' ', '') ||
        COALESCE(s.department_4 || ' ', '') ||
        COALESCE(s.department_3 || ' ', '') ||
        COALESCE(s.department_2 || ' ', '') ||
        COALESCE(s.department_1, '')
    ) AS full_position_name
FROM kpi_staff_structure AS s
WHERE s.department_1 = 'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге'
  AND s.department_2 = 'Управление добровольного медицинского страхования'
  AND s.department_3 IS NULL
  AND s.department_4 IS NULL
  AND s.department_5 IS NULL
  AND s.department_6 IS NULL
  AND s.department_7 IS NULL
  AND s.department_8 IS NULL
ORDER BY s.position, s.staff_structure_id;
```

Пример запроса для получения всех позиций, работающих в подразделениях управления
`Управление добровольного медицинского страхования Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге`.
В этом запросе выбираются только дочерние подразделения управления:

```sql
SELECT
    s.staff_structure_id,
    s.department_3,
    s.department_4,
    s.department_5,
    s.department_6,
    s.department_7,
    s.department_8,
    s.employee_group,
    s.position,
    TRIM(
        s.position || ' ' ||
        '(' || s.employee_group || ') ' ||
        COALESCE(s.department_8 || ' ', '') ||
        COALESCE(s.department_7 || ' ', '') ||
        COALESCE(s.department_6 || ' ', '') ||
        COALESCE(s.department_5 || ' ', '') ||
        COALESCE(s.department_4 || ' ', '') ||
        COALESCE(s.department_3 || ' ', '') ||
        COALESCE(s.department_2 || ' ', '') ||
        COALESCE(s.department_1, '')
    ) AS full_position_name
FROM kpi_staff_structure AS s
WHERE s.department_1 = 'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге'
  AND s.department_2 = 'Управление добровольного медицинского страхования'
  AND (
      s.department_3 IS NOT NULL
      OR s.department_4 IS NOT NULL
      OR s.department_5 IS NOT NULL
      OR s.department_6 IS NOT NULL
      OR s.department_7 IS NOT NULL
      OR s.department_8 IS NOT NULL
  )
ORDER BY
    s.department_3,
    s.department_4,
    s.department_5,
    s.department_6,
    s.department_7,
    s.department_8,
    s.employee_group,
    s.position,
    s.staff_structure_id;
```

## Таблица `kpi_values`

Таблица фиксирует назначение KPI на конкретный организационно-ролевой контекст.
Одна запись означает, что указанный KPI применим к связанной строке штатной структуры, центру ответственности и уровню должности.
Через `staff_structure_ref` запись раскрывается до подразделений, группы работников и должности, а через `kpi_method_ref` - до методики расчета KPI.

### Поля

- `source_row`
  Технический идентификатор записи назначения KPI.
  Первичный ключ таблицы.

- `responsibility_center`
  Центр ответственности.
  Примеры: центр дохода, центр промежуточной прибыли, центр функционального сопровождения.

- `position_group`
  Уровень должности для назначения KPI.
  Например: `Руководитель верхнего звена`, `Руководитель среднего звена`, `Исполнители`.
  Не путай это поле с `kpi_staff_structure.employee_group`: `employee_group` означает группу работников внутри подразделения.

- `staff_structure_ref`
  Ссылка на строку штатной структуры в `kpi_staff_structure.staff_structure_id`.
  Через это поле получаются `department_1` ... `department_8`, `employee_group` и `position`.

- `kpi_name`
  Наименование KPI в контексте назначения.
  Для связи с методикой расчета не используй это поле напрямую; используй `kpi_method_ref`.
  Если задача - найти все KPI вместе с соответствующей методикой, используй это поле для поиска в сочетании с полями таблицы `kpi_method`, например `kpi_method.kpi_name`, `kpi_method.calculation_method` и `kpi_method.note`.

- `kpi_method_ref`
  Ссылка на строку методики расчета в `kpi_method.source_row`.
  Может быть `NULL`, если для назначения KPI нет связанной методики.

- `calculation_detail`
  Уровень детализации расчета KPI для конкретного назначения.
  Поле показывает, на каком объекте или в каком периметре должен считаться показатель: по всей компании, филиалу, отделу, сектору, операционному офису, работнику и т.п.
  Это не формула расчета и не методика; формула и пояснения находятся в `kpi_method.calculation_method` и `kpi_method.note`.
  В текущей базе поле заполнено во всех строках `kpi_values`.

  Текущие значения:

  - `по Компании` - расчет на уровне компании в целом.
  - `по филиалу` - расчет на уровне филиала в целом.
  - `по филиалу (по группе работников)` - расчет на уровне филиала, но в разрезе группы работников.
  - `по РС` - расчет в периметре РС, как отдельного бизнес-/организационного периметра данных.
  - `по отделу` - расчет на уровне отдела.
  - `по сектору` - расчет на уровне сектора.
  - `по опер.офису` - расчет на уровне операционного офиса.
  - `по работнику` - расчет на уровне конкретного работника.
  - `по курируемым МП` - расчет по периметру курируемых МП.

- `business_line`
  Линия бизнеса.

- `pool`
  Флаг участия KPI в пуле.
  Допустимые значения: `да` и `нет`.
  `да` означает, что назначение KPI относится к пулу; `нет` означает, что не относится.
  Если значение не задано, при импорте записывается `нет`.

- `other_analytics`
  Прочая аналитика для назначения KPI.

- `calculation_frequency`
  Периодичность расчета KPI.

- `calculation_specifics`
  Специфика расчета KPI.

- `analytics_1`
  Дополнительная аналитика 1.
  В текущих данных полностью пустая.

- `analytics_2`
  Дополнительная аналитика 2.
  В текущих данных полностью пустая.

- `analytics_3`
  Дополнительная аналитика 3.
  В текущих данных полностью пустая.

- `analytics_4`
  Дополнительная аналитика 4.
  В текущих данных полностью пустая.

### Как использовать таблицу `kpi_values`

- Для анализа KPI по подразделениям всегда присоединяй `kpi_staff_structure`.
- Для получения методики расчета присоединяй `kpi_method` по `kpi_method_ref`.
- Не группируй только по `kpi_name`, если нужен точный набор методик: одинаковые или похожие KPI могут иметь различия в сносках и формулировках.
- Для анализа по ролям используй `kpi_staff_structure.employee_group` как группу работников внутри подразделения и `kpi_values.position_group` как уровень должности.
- Для анализа по организационной структуре используй поля `department_*`, `employee_group` и `position` из `kpi_staff_structure`.
- Для понимания уровня агрегации KPI используй `calculation_detail`: например, один и тот же KPI может считаться `по работнику`, `по отделу` или `по филиалу`.

Пример запроса для KPI с подразделением, должностью и методикой:

```sql
SELECT
    v.source_row,
    s.department_1,
    s.department_2,
    s.department_3,
    s.employee_group,
    s.position,
    v.responsibility_center,
    v.position_group,
    v.kpi_name,
    m.calculation_method,
    m.note
FROM kpi_values AS v
JOIN kpi_staff_structure AS s
    ON s.staff_structure_id = v.staff_structure_ref
LEFT JOIN kpi_method AS m
    ON m.source_row = v.kpi_method_ref
ORDER BY v.source_row;
```

Пример поиска KPI по конкретной полной позиции:

```sql
SELECT
    v.source_row,
    v.kpi_name,
    v.responsibility_center,
    v.position_group,
    s.employee_group,
    s.department_1,
    s.department_2,
    s.department_3,
    s.position,
    m.calculation_method,
    m.note
FROM kpi_values AS v
JOIN kpi_staff_structure AS s
    ON s.staff_structure_id = v.staff_structure_ref
LEFT JOIN kpi_method AS m
    ON m.source_row = v.kpi_method_ref
WHERE s.position = 'Начальник отдела'
  AND s.employee_group = 'Сопровождение функциональное (РС)'
  AND s.department_2 = 'Отдел информационных технологий'
  AND s.department_1 = 'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге'
ORDER BY v.kpi_name, v.source_row;
```

Пример построения полного наименования строки назначения KPI:

```sql
SELECT
    v.source_row,
    CASE v.position_group
        WHEN 'Исполнители' THEN 'исполнитель'
        WHEN 'Руководитель среднего звена' THEN 'руководитель среднего звена'
        WHEN 'Руководитель верхнего звена' THEN 'руководитель верхнего звена'
        ELSE v.position_group
    END || ': ' ||
    s.position || ' ' ||
    '(' || s.employee_group || ') ' ||
    COALESCE(s.department_8 || ' ', '') ||
    COALESCE(s.department_7 || ' ', '') ||
    COALESCE(s.department_6 || ' ', '') ||
    COALESCE(s.department_5 || ' ', '') ||
    COALESCE(s.department_4 || ' ', '') ||
    COALESCE(s.department_3 || ' ', '') ||
    COALESCE(s.department_2 || ' ', '') ||
    COALESCE(s.department_1, '') AS full_assignment_name
FROM kpi_values AS v
JOIN kpi_staff_structure AS s
    ON s.staff_structure_id = v.staff_structure_ref
WHERE v.source_row = 182;
```

## Таблица `kpi_method`

Таблица является справочником методик расчета KPI.
Одна запись описывает методику расчета и примечания для KPI, на которую могут ссылаться назначения KPI из `kpi_values`.

### Поля

- `source_row`
  Технический идентификатор записи методики расчета.
  Первичный ключ таблицы.
  Используется как цель ссылки из `kpi_values.kpi_method_ref`.

- `kpi_name`
  Наименование KPI в справочнике методик.
  Может отличаться от `kpi_values.kpi_name` из-за сносок, пробелов или уточнений.

- `calculation_method`
  Текст методики расчета KPI.
  Может содержать многострочный текст.

- `note`
  Примечание к методике расчета.
  Может быть `NULL`.

### Как использовать таблицу `kpi_method`

- Используй таблицу для объяснения того, как рассчитывается KPI.
- Присоединяй таблицу к `kpi_values` через `kpi_method_ref`.
- Если `kpi_method_ref` пустой, не пытайся автоматически подбирать методику по `kpi_name` без дополнительной проверки: названия могут быть неоднозначными.
- `source_row` полезен как технический ключ методики расчета.

Пример запроса для KPI без связанной методики:

```sql
SELECT
    v.source_row,
    v.kpi_name,
    v.responsibility_center,
    v.position_group,
    s.department_1,
    s.department_2,
    s.employee_group,
    s.position
FROM kpi_values AS v
JOIN kpi_staff_structure AS s
    ON s.staff_structure_id = v.staff_structure_ref
WHERE v.kpi_method_ref IS NULL
ORDER BY v.source_row;
```

## Индексы и ограничения

- `kpi_method`
  Первичный ключ: `source_row`.

- `kpi_staff_structure`
  Первичный ключ: `staff_structure_id`.
  Уникальный индекс `idx_kpi_staff_structure_unique` по:
  `department_1`, `department_2`, `department_3`, `department_4`,
  `department_5`, `department_6`, `department_7`, `department_8`, `employee_group`, `position`.

- `kpi_values`
  Первичный ключ: `source_row`.
  Индекс `idx_kpi_values_staff_structure_ref` по `staff_structure_ref`.
  Индекс `idx_kpi_values_kpi_method_ref` по `kpi_method_ref`.
  Индекс `idx_kpi_values_kpi_name` по `kpi_name`.

Внешние ключи:

- `kpi_values.staff_structure_ref` -> `kpi_staff_structure.staff_structure_id`.
- `kpi_values.kpi_method_ref` -> `kpi_method.source_row`.

## Практические рекомендации для запросов к базе

1. Если вопрос про штатное расписание, подразделения, группы работников или должности, начинай с `kpi_staff_structure`.

2. Если вопрос про назначенные KPI, начинай с `kpi_values`.

3. Если вопрос про методику расчета KPI, используй `kpi_method`, но присоединяй ее через `kpi_values.kpi_method_ref`.

4. Если нужно показать полный контекст KPI, обычно нужны все три таблицы:
   `kpi_values` + `kpi_staff_structure` + `kpi_method`.

5. Не делай внутренний `JOIN` к `kpi_method`, если нужно сохранить все строки KPI.
   Используй `LEFT JOIN`, потому что часть строк не имеет методики.

6. Не ищи подразделения и группу работников в `kpi_values`: после нормализации они находятся только в `kpi_staff_structure`.

7. Не считай `NULL` в `department_2` ... `department_8` ошибкой.
   Это означает, что соответствующий уровень организационной структуры не задан.

8. Для пользовательского вывода лучше показывать исходные текстовые поля как есть.
   База не создает дополнительных lowercase- или nocase-полей для поиска.

9. Для фильтрации по тексту в SQLite можно использовать `LIKE`, но учитывай, что русскоязычный регистр лучше обрабатывать осторожно на стороне приложения или отдельной нормализацией, если она появится позже.
