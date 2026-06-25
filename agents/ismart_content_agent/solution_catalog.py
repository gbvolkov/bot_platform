from __future__ import annotations

from textwrap import indent
from typing import Any

from .contracts import ExecutableSolution, TemplateId


TaskKey = tuple[int, str, int]


CONTROL_ANSWERS: dict[TaskKey, str] = {
    (2, "L1", 1): "Пропущена закрывающая кавычка.",
    (2, "L1", 2): "Неверно написано имя prnt; правильно: print.",
}


PYTHON_SOLUTIONS: dict[TaskKey, tuple[str, list[tuple[str, str]]]] = {
    (1, "L1", 1): ('print("Привет!")', [("", "Привет!\n")]),
    (1, "L1", 2): ('print("Меня зовут Анна")', [("", "Меня зовут Анна\n")]),
    (1, "L2", 3): ('print("Привет")', [("", "Привет\n")]),
    (1, "L2", 4): ('print("Привет")', [("", "Привет\n")]),
    (1, "L2", 5): ('print("Привет")', [("", "Привет\n")]),
    (2, "L2", 3): ('print("Привет")', [("", "Привет\n")]),
    (2, "L2", 4): ('print("Привет")', [("", "Привет\n")]),
    (2, "L2", 5): ('print("Привет")', [("", "Привет\n")]),
    (3, "L1", 1): ('name = "Анна"\nprint(name)', [("", "Анна\n")]),
    (3, "L1", 2): ('age = 14\nprint(age)', [("", "14\n")]),
    (3, "L2", 3): ('name = "Анна"\nage = 14\nprint(name, age)', [("", "Анна 14\n")]),
    (3, "L2", 4): ('name = "Анна"\nprint("Меня зовут", name)', [("", "Меня зовут Анна\n")]),
    (3, "L2", 5): ('color = "синий"\nanimal = "кот"\nprint(color, animal)', [("", "синий кот\n")]),
    (4, "L1", 1): ('input()', [("Привет\n", "")]),
    (4, "L1", 2): ('name = input("Как тебя зовут? ")\nprint("Привет,", name)', [("Аня\n", "Как тебя зовут? Привет, Аня\n")]),
    (4, "L2", 3): ('age = int(input())\nprint("Через 5 лет тебе будет", age + 5)', [("14\n", "Через 5 лет тебе будет 19\n")]),
    (4, "L2", 4): ('a = int(input())\nb = int(input())\nprint(a + b)', [("3\n4\n", "7\n")]),
    (4, "L2", 5): ('number = int(input())\nprint(number * 5)', [("8\n", "40\n")]),
    (5, "L1", 1): ('a = int(input())\nprint(a * 2)', [("7\n", "14\n")]),
    (5, "L1", 2): ('a = int(input())\nb = int(input())\nprint(a + b)\nprint(a - b)', [("9\n4\n", "13\n5\n")]),
    (5, "L2", 3): ('minutes = int(input())\nprint(f"{minutes // 60} часа {minutes % 60} минут")', [("130\n", "2 часа 10 минут\n")]),
    (5, "L2", 4): ('apples = int(input())\nchildren = int(input())\nprint(apples // children)\nprint(apples % children)', [("11\n5\n", "2\n1\n")]),
    (5, "L2", 5): ('money = int(input())\nprice = int(input())\nprint(money // price)\nprint(money % price)', [("1000\n120\n", "8\n40\n")]),
    (5, "L2", 6): ('a = int(input())\nb = int(input())\nc = int(input())\nprint((a + b) * c)', [("2\n3\n4\n", "20\n")]),
    (6, "L1", 1): ('price = 99.99\nprint(price)', [("", "99.99\n")]),
    (6, "L1", 2): ('value = float(input())\nprint(int(value))', [("7.5\n", "7\n")]),
    (6, "L2", 3): ('price = float(input())\nquantity = int(input())\nprint(round(price * quantity, 2))', [("99.99\n3\n", "299.97\n")]),
    (6, "L2", 4): ('value = float(input())\nprint(round(value))', [("7.5\n", "8\n")]),
    (6, "L2", 5): ('a = float(input())\nb = float(input())\nprint(a + b)', [("0.1\n0.2\n", "0.30000000000000004\n")]),
    (6, "L2", 6): ('rubles = float(input())\nkopecks = int(round(rubles * 100))\nprint(kopecks)\nprint(kopecks / 100)', [("123.45\n", "12345\n123.45\n")]),
    (7, "L1", 1): ('hours = int(input())\nprint(hours * 60)', [("3\n", "180\n")]),
    (7, "L1", 2): ('minutes = int(input())\nprint(minutes * 60)', [("7\n", "420\n")]),
    (7, "L2", 3): ('seconds = int(input())\nprint(seconds // 60)\nprint(seconds % 60)', [("3665\n", "61\n5\n")]),
    (7, "L2", 4): ('seconds = int(input())\nhours = seconds // 3600\nminutes = seconds % 3600 // 60\nrest = seconds % 60\nprint(hours, minutes, rest)', [("3665\n", "1 1 5\n")]),
    (7, "L2", 5): ('meters = int(input())\nprint(meters // 1000, meters % 1000)', [("3665\n", "3 665\n")]),
    (7, "L2", 6): ('price = int(input())\nmoney = int(input())\ncount = money // price\nprint(count)\nprint(money - count * price)', [("120\n1000\n", "8\n40\n")]),
    (8, "L1", 1): ('price = float(input())\nprint(round(price * 1.1, 2))', [("100\n", "110.0\n")]),
    (8, "L1", 2): ('value = float(input())\nprint(round(value))', [("7.5\n", "8\n")]),
    (8, "L2", 3): ('price = float(input())\ndiscount = int(input())\nprint(round(price * (1 - discount / 100), 2))', [("1000\n15\n", "850.0\n")]),
    (8, "L2", 4): ('rubles = float(input())\nprint(int(round(rubles * 100)))', [("123.45\n", "12345\n")]),
    (8, "L2", 5): ('kopecks = int(input())\nprint(kopecks / 100)', [("12345\n", "123.45\n")]),
    (8, "L2", 6): ('a = float(input())\nb = float(input())\nc = float(input())\ntotal = a + b + c\nprint(round(total))\nprint(round(total, 1))', [("1.2\n2.3\n3.4\n", "7\n6.9\n")]),
    (9, "L1", 1): ('name = "Анна"\nprint(len(name))', [("", "4\n")]),
    (9, "L1", 2): ('first = "Анна"\nlast = "Иванова"\nprint(first + " " + last)', [("", "Анна Иванова\n")]),
    (9, "L2", 3): ('name = input()\nlast = input()\nprint(last + " " + name)', [("Анна\nИванова\n", "Иванова Анна\n")]),
    (9, "L2", 4): ('word = input()\nprint(len(word))\nprint(" ".join([word] * 3))', [("кот\n", "3\nкот кот кот\n")]),
    (9, "L2", 5): ('text = input()\nprint(text[0])', [("Python\n", "P\n")]),
    (9, "L2", 6): ('word = input()\nprint(word[0], word[-1])', [("кот\n", "к т\n")]),
}


def batch_task_configuration(task: dict[str, Any]) -> tuple[TemplateId, dict[str, Any]]:
    key = (task["lesson_number"], task["task_level"], task["task_number"])
    if key in CONTROL_ANSWERS:
        return "control_question", {"answer_key": CONTROL_ANSWERS[key]}
    if key not in PYTHON_SOLUTIONS:
        raise ValueError(f"No executable solution registered for source task {key}")
    body, test_cases = PYTHON_SOLUTIONS[key]
    tests = [
        {
            "test_id": f"lesson-{key[0]}-{key[1].lower()}-{key[2]}-test-{index}",
            "stdin": stdin,
            "expected_stdout": expected,
        }
        for index, (stdin, expected) in enumerate(test_cases, start=1)
    ]
    return "practice_python", {
        "service_solution": ExecutableSolution(code=_script(body)).model_dump(mode="json"),
        "tests": tests,
    }


def _script(body: str) -> str:
    return (
        "def main():\n"
        + indent(body, "    ")
        + "\n\nif __name__ == \"__main__\":\n"
        + "    main()\n"
    )
