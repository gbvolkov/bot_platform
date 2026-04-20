import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langchain_core.documents import Document

from utils.kpi_dialog_engine import KPIDialogEngine
from utils.kpi_retrieval import KPIHybridRetriever, build_kpi_catalog


def _row_doc(department_path: str, position: str, kpi_name: str, row: int) -> Document:
    page_content = "\n".join(
        [
            "Тип записи: KPI строка матрицы",
            "Источник файла: fixture.xlsx",
            "Лист: Филиал в г.СПб",
            f"Строка Excel: {row}",
            f"Путь подразделения: {department_path}",
            f"Контур роли: Центр дохода | Продавцы | Руководитель среднего звена | {position}",
            "Центр ответственности: Центр дохода",
            "Группа работников: Продавцы",
            "Группа должностей: Руководитель среднего звена",
            f"Должность: {position}",
            f"KPI: {kpi_name}",
            "Детализация расчета: по филиалу",
            "Периодичность расчета: Квартал",
            "Специфика расчета: -",
        ]
    )
    return Document(
        page_content=page_content,
        metadata={
            "source": "fixture.xlsx",
            "sheet_name": "Филиал в г.СПб",
            "excel_doc_type": "kpi_row",
            "department_path": department_path,
            "center": "Центр дохода",
            "position": position,
            "role_scope": f"Центр дохода | Продавцы | Руководитель среднего звена | {position}",
            "kpi_name": kpi_name,
            "sheet_row": row,
        },
    )


class KPIDialogEngineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        row_docs = [
            _row_doc(
                'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге > Отдел по работе с банками',
                "Начальник отдела",
                "Полученная страховая премия",
                38,
            ),
            _row_doc(
                'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге > Отдел по работе с банками',
                "Начальник отдела",
                "Нетто-комиссионное вознаграждение",
                39,
            ),
            _row_doc(
                'Филиал СПАО "Ингосстрах" в г. Санкт-Петербурге > Отдел по работе с автодилерами',
                "Начальник отдела",
                "Доля пользователей мобильного приложения",
                17,
            ),
        ]
        catalog = build_kpi_catalog(row_docs=row_docs, methodology_docs=[])
        retriever = KPIHybridRetriever(
            catalog=catalog,
            semantic_search=lambda query, n_results, where=None: [],
        )
        cls.engine = KPIDialogEngine(catalog=catalog, retriever=retriever)

    def test_resolve_state_keeps_position_and_requests_department_options(self) -> None:
        state = self.engine.dialog_resolver.resolve_state(
            "я не помню подразделение, какие есть?",
            [
                ("я не помню подразделение, какие есть?", "user", "2026-03-23 12:04:00"),
                (
                    "Для должности «Начальник отдела» KPI зависят от подразделения. Уточните, пожалуйста, подразделение.",
                    "assistant",
                    "2026-03-23 12:03:00",
                ),
                ("какие у меня kpi?", "user", "2026-03-23 12:02:00"),
                ("я начальник отдела", "user", "2026-03-23 12:01:00"),
            ],
        )

        self.assertEqual(state.position, "Начальник отдела")
        self.assertEqual(state.department, None)
        self.assertEqual(state.requested_slot, "department")
        self.assertEqual(state.intent, "list_slot_options")

    def test_handle_lists_department_options_for_known_position(self) -> None:
        result = self.engine.handle(
            "я не помню подразделение, какие есть?",
            [
                ("я не помню подразделение, какие есть?", "user", "2026-03-23 12:04:00"),
                (
                    "Для должности «Начальник отдела» KPI зависят от подразделения. Уточните, пожалуйста, подразделение.",
                    "assistant",
                    "2026-03-23 12:03:00",
                ),
                ("какие у меня kpi?", "user", "2026-03-23 12:02:00"),
                ("я начальник отдела", "user", "2026-03-23 12:01:00"),
            ],
        )

        self.assertIn("Зафиксировал: должность", result.direct_answer)
        self.assertIn("Отдел по работе с банками", result.direct_answer)
        self.assertIn("Отдел по работе с автодилерами", result.direct_answer)

    def test_handle_recalls_known_position(self) -> None:
        result = self.engine.handle(
            "какая у меня должность?",
            [
                ("какая у меня должность?", "user", "2026-03-23 12:04:00"),
                ("какие у меня kpi?", "user", "2026-03-23 12:03:00"),
                ("я начальник отдела", "user", "2026-03-23 12:01:00"),
            ],
        )

        self.assertEqual(result.direct_answer, "Ваша должность — «Начальник отдела».")

    def test_handle_resolves_kpi_after_department_follow_up(self) -> None:
        result = self.engine.handle(
            "отдел работы с банками",
            [
                ("отдел работы с банками", "user", "2026-03-23 12:05:00"),
                ("какая у меня должность?", "user", "2026-03-23 12:04:00"),
                ("я не помню подразделение, какие есть?", "user", "2026-03-23 12:03:00"),
                ("какие у меня kpi?", "user", "2026-03-23 12:02:00"),
                ("я начальник отдела", "user", "2026-03-23 12:01:00"),
            ],
        )

        self.assertIn("Для должности «Начальник отдела» в «Отдел по работе с банками» применимы следующие KPI:", result.direct_answer)
        self.assertIn("Полученная страховая премия", result.direct_answer)
        self.assertIn("Нетто-комиссионное вознаграждение", result.direct_answer)

    def test_handle_requests_specific_kpi_for_freeform_methodology_follow_up(self) -> None:
        result = self.engine.handle(
            "покажи методику расчета",
            [
                ("покажи методику расчета", "user", "2026-03-23 12:06:00"),
                (
                    "Для должности «Начальник отдела» в «Отдел по работе с банками» применимы следующие KPI:\n\n"
                    "1. Полученная страховая премия\n"
                    "2. Нетто-комиссионное вознаграждение\n\n"
                    "Если нужно, могу отдельно показать методику расчета по любому KPI из списка.",
                    "assistant",
                    "2026-03-23 12:05:00",
                ),
            ],
        )

        self.assertIn("по какому KPI показать методику расчета", result.direct_answer)
        self.assertIn("Можно назвать номер или название", result.direct_answer)

    def test_handle_requests_specific_kpi_for_short_affirmative_follow_up(self) -> None:
        result = self.engine.handle(
            "нужно",
            [
                ("нужно", "user", "2026-03-23 12:06:00"),
                (
                    "Для должности «Начальник отдела» в «Отдел по работе с банками» применимы следующие KPI:\n\n"
                    "1. Полученная страховая премия\n"
                    "2. Нетто-комиссионное вознаграждение\n\n"
                    "Если нужно, могу отдельно показать методику расчета по любому KPI из списка.",
                    "assistant",
                    "2026-03-23 12:05:00",
                ),
            ],
        )

        self.assertIn("по какому KPI показать методику расчета", result.direct_answer)

    def test_handle_resolves_single_kpi_for_short_affirmative_follow_up(self) -> None:
        result = self.engine.handle(
            "нужно",
            [
                ("нужно", "user", "2026-03-23 12:06:00"),
                (
                    "Для должности «Начальник отдела» в «Отдел по работе с банками» применимы следующие KPI:\n\n"
                    "1. Нетто-комиссионное вознаграждение\n\n"
                    "Если нужно, могу отдельно показать методику расчета по любому KPI из списка.",
                    "assistant",
                    "2026-03-23 12:05:00",
                ),
            ],
        )

        self.assertEqual(result.direct_answer, "")
        self.assertTrue(result.documents)
        self.assertTrue(
            any(doc.metadata.get("kpi_name") == "Нетто-комиссионное вознаграждение" for doc in result.documents)
        )


    def test_handle_aggregate_catalog_count(self) -> None:
        result = self.engine.handle(
            "сколько всего уникальных KPI ты знаешь?",
            [("сколько всего уникальных KPI ты знаешь?", "user", "2026-04-15 09:00:00")],
        )
        self.assertIn("уникальных KPI", result.direct_answer)
        self.assertIn("3", result.direct_answer)  # три уникальных KPI в фикстуре

    def test_handle_aggregate_catalog_full_list(self) -> None:
        result = self.engine.handle(
            "покажи полный список всех KPI филиала",
            [("покажи полный список всех KPI филиала", "user", "2026-04-15 09:00:00")],
        )
        self.assertIn("Полученная страховая премия", result.direct_answer)
        self.assertIn("Нетто-комиссионное вознаграждение", result.direct_answer)
        self.assertIn("Доля пользователей мобильного приложения", result.direct_answer)

    def test_handle_reverse_kpi_lookup(self) -> None:
        result = self.engine.handle(
            "у кого есть KPI Нетто-комиссионное вознаграждение?",
            [("у кого есть KPI Нетто-комиссионное вознаграждение?", "user", "2026-04-15 09:00:00")],
        )
        self.assertIn("Нетто-комиссионное вознаграждение", result.direct_answer)
        self.assertIn("Отдел по работе с банками", result.direct_answer)

    def test_handle_reverse_kpi_lookup_missing_kpi(self) -> None:
        result = self.engine.handle(
            "у кого есть KPI Коэффициент промежуточной маржи?",
            [("у кого есть KPI Коэффициент промежуточной маржи?", "user", "2026-04-15 09:00:00")],
        )
        self.assertIn("Коэффициент промежуточной маржи", result.direct_answer)
        self.assertIn("не зафиксирован", result.direct_answer)

    def test_handle_kpi_membership_positive(self) -> None:
        result = self.engine.handle(
            "а есть у него показатель Нетто-комиссионное вознаграждение?",
            [
                (
                    "а есть у него показатель Нетто-комиссионное вознаграждение?",
                    "user",
                    "2026-04-15 09:05:00",
                ),
                ("Отдел по работе с банками", "user", "2026-04-15 09:04:00"),
                ("я начальник отдела", "user", "2026-04-15 09:03:00"),
            ],
        )
        self.assertTrue(
            result.direct_answer.startswith("Да,"),
            msg=f"unexpected: {result.direct_answer!r}",
        )
        self.assertIn("Нетто-комиссионное вознаграждение", result.direct_answer)

    def test_handle_kpi_membership_negative(self) -> None:
        result = self.engine.handle(
            "а есть у него показатель Доля пользователей мобильного приложения?",
            [
                (
                    "а есть у него показатель Доля пользователей мобильного приложения?",
                    "user",
                    "2026-04-15 09:05:00",
                ),
                ("Отдел по работе с банками", "user", "2026-04-15 09:04:00"),
                ("я начальник отдела", "user", "2026-04-15 09:03:00"),
            ],
        )
        self.assertTrue(
            result.direct_answer.startswith("Нет,"),
            msg=f"unexpected: {result.direct_answer!r}",
        )


if __name__ == "__main__":
    unittest.main()
