from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NotRequired, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .attempts import AttemptArtifactStore
from .context import (
    build_current_control_autocheck_prompt,
    build_current_control_autocheck_system_prompt,
    build_generation_prompt,
    build_generator_system_prompt,
    build_intermediate_assessment_artifact_prompt,
    build_intermediate_assessment_artifact_system_prompt,
    build_package_validation_prompt,
    build_practice_template_prompt,
    build_practice_template_system_prompt,
    build_practice_variant_prompt,
    build_practice_variant_system_prompt,
    build_self_work_autocheck_prompt,
    build_self_work_autocheck_system_prompt,
    build_validation_controller_prompt,
    build_validation_controller_system_prompt,
    build_validation_prompt,
    build_validator_system_prompt,
    source_contract_for_spec,
)
from .contracts import (
    IsmartGenerationConfig,
    MaterialResult,
    MaterialSpec,
    ReferenceBundle,
    ValidationResult,
)
from .schemas import (
    CurrentControlAutocheckSet,
    GeneratedMaterial,
    IntermediateAssessmentArtifact,
    MaterialValidationDecision,
    PackageValidationDecision,
    PracticeTaskInstanceSet,
    PracticeTaskTemplateSet,
    SelfWorkAutocheckSet,
    ValidationControllerDecision,
)
from .sources import read_prompt_files
from .trace import TraceLogger
from .validators import RuleValidator


HTML_TEMPLATE_PLACEHOLDER = "{{ body_html }}"
HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "cc_lesson_template.html"


@dataclass(frozen=True)
class ContentBoundary:
    raw_content: str
    content: str
    issues: list[str]
    prefix: str
    tail: str


@dataclass(frozen=True)
class GeneratedAttempt:
    raw_content: str
    content: str
    boundary_issues: list[str]
    agent_notes: list[str]
    generation_artifacts: dict[str, Any]
    structural_validation: ValidationResult


@dataclass(frozen=True)
class HtmlFormatTemplate:
    template_html: str
    style_block: str

    def render(self, body_html: str) -> str:
        return self.template_html.replace(HTML_TEMPLATE_PLACEHOLDER, body_html)


def load_html_format_template(template_path: Path = HTML_TEMPLATE_PATH) -> HtmlFormatTemplate:
    if not template_path.exists():
        raise FileNotFoundError(f"HTML template file not found: {template_path}")
    template_html = template_path.read_text(encoding="utf-8").strip()
    if HTML_TEMPLATE_PLACEHOLDER not in template_html:
        raise ValueError(f"HTML template must contain {HTML_TEMPLATE_PLACEHOLDER}")
    if '<div class="cc-lesson">' not in template_html or not template_html.endswith("</div>"):
        raise ValueError(f"HTML template must wrap content in <div class=\"cc-lesson\">...</div>: {template_path}")
    match = re.search(r"<style>.*?</style>", template_html, flags=re.DOTALL)
    if not match:
        raise ValueError(f"HTML template must start with a canonical <style> block: {template_path}")
    if template_html.find(match.group(0)) != 0:
        raise ValueError(f"HTML template must start with <style>: {template_path}")
    return HtmlFormatTemplate(template_html=template_html, style_block=match.group(0).strip())


def isolate_material_html(raw_content: str) -> ContentBoundary:
    raw = raw_content.strip()
    issues: list[str] = []
    prefix = ""
    tail = ""
    content = raw

    lower = content.lower()
    style_index = lower.find("<style>")
    if style_index > 0:
        prefix = content[:style_index].strip()
        issues.append("generated content had non-HTML text before <style>; prefix was stripped before validation")
        content = content[style_index:].strip()
        lower = content.lower()

    div_end_index = lower.rfind("</div>")
    if div_end_index >= 0:
        html_end = div_end_index + len("</div>")
        tail = content[html_end:].strip()
        if tail:
            snippet = tail[:240].replace("\n", "\\n")
            issues.append(f"generated content had non-HTML tail after final </div>; tail was stripped before validation: {snippet}")
            content = content[:html_end].strip()

    return ContentBoundary(raw_content=raw, content=content, issues=issues, prefix=prefix, tail=tail)


def _normalize_copy_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip().lower()


def _normalize_test_items(tests: Any) -> list[Any]:
    if not isinstance(tests, list):
        return []
    normalized_tests: list[Any] = []
    for test in tests:
        if not isinstance(test, dict):
            normalized_tests.append(test)
            continue
        normalized = {str(key): str(value) for key, value in test.items()}
        if "input" not in normalized:
            for alias in ("stdin", "in", "вход"):
                if alias in normalized:
                    normalized["input"] = normalized[alias]
                    break
        if "expected_output" not in normalized:
            for alias in ("output", "stdout", "expected", "result", "ожидаемый_вывод"):
                if alias in normalized:
                    normalized["expected_output"] = normalized[alias]
                    break
        normalized_tests.append(normalized)
    return normalized_tests


def _normalize_practice_instance_tests(instances: dict[str, Any]) -> dict[str, Any]:
    tasks = instances.get("tasks")
    if not isinstance(tasks, list):
        return instances
    for task_item in tasks:
        if not isinstance(task_item, dict):
            continue
        tests = _normalize_test_items(task_item.get("tests"))
        runtime_tests = _normalize_test_items(task_item.get("runtime_tests"))
        if not runtime_tests and tests:
            runtime_tests = list(tests)
        if not tests and runtime_tests:
            tests = list(runtime_tests)
        task_item["tests"] = tests
        task_item["runtime_tests"] = runtime_tests
    return instances


def _model_to_dict(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value.dict()


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, BaseModel):
            result.append(_model_to_dict(item))
    return result


def _plain_text(value: Any) -> str:
    return str(value or "").strip()


def _html_code(value: Any) -> str:
    return html.escape(str(value or "").rstrip("\n"))


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _plain_text(value)
        if text:
            return text
    return ""


def _test_input(test: Mapping[str, Any]) -> str:
    return _first_nonempty(test.get("input"), test.get("stdin"), test.get("in"), test.get("вход"))


def _test_expected(test: Mapping[str, Any]) -> tuple[str, str]:
    expected_output = _first_nonempty(
        test.get("expected_output"),
        test.get("stdout"),
        test.get("output"),
        test.get("expected"),
        test.get("result"),
        test.get("ожидаемый_вывод"),
    )
    if expected_output:
        return "Ожидаемый вывод", expected_output
    expected_error = _first_nonempty(test.get("expected_error"), test.get("error_message"), test.get("stderr"))
    if expected_error:
        return "Ожидаемая ошибка", expected_error
    return "Ожидаемый результат", _first_nonempty(test.get("expected_result"), test.get("check"), "не указан")


def _render_practice_tests(tests: list[dict[str, Any]]) -> str:
    if not tests:
        return ""
    rows = []
    for index, test in enumerate(tests, start=1):
        label, expected = _test_expected(test)
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><pre><code>{_html_code(_test_input(test))}</code></pre></td>"
            f"<td>{html.escape(label)}</td>"
            f"<td><pre><code>{_html_code(expected)}</code></pre></td>"
            "</tr>"
        )
    return (
        "<p><strong>Проверка на тестах:</strong></p>"
        '<table class="cc-table"><thead><tr>'
        "<th>#</th><th>Вход</th><th>Тип результата</th><th>Ожидаемый результат</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_practice_manual_checks(checks: list[Any]) -> str:
    visible = [_plain_text(item) for item in checks if _plain_text(item)]
    if not visible:
        return ""
    items = "".join(f"<li>{html.escape(item)}</li>" for item in visible)
    return f"<p><strong>Как проверить вручную:</strong></p><ul>{items}</ul>"


def _render_practice_subtasks(subtasks: list[Any]) -> str:
    rows: list[str] = []
    for item in subtasks:
        if not isinstance(item, Mapping):
            text = _plain_text(item)
            if text:
                rows.append(f"<li>{html.escape(text)}</li>")
            continue
        text = " — ".join(_plain_text(value) for value in item.values() if _plain_text(value))
        if text:
            rows.append(f"<li>{html.escape(text)}</li>")
    if not rows:
        return ""
    return "<p><strong>Подзадачи:</strong></p><ol>" + "".join(rows) + "</ol>"


def _practice_lesson_heading(lesson: Mapping[str, Any]) -> str:
    lesson_title = _first_nonempty(lesson.get("title"), lesson.get("name"), "Задания Python")
    lesson_number = _first_nonempty(lesson.get("lesson_number"), lesson.get("number"), lesson.get("№"))
    if lesson_number:
        return f"Занятие {lesson_number}. {lesson_title}"
    return lesson_title


def _practice_task_needs_empty_code_block(item: Mapping[str, Any]) -> bool:
    run_mode = _normalize_copy_text(str(item.get("run_mode") or ""))
    if run_mode in {"manual_only", "needs_platform_clarification"}:
        return False

    task_type = _normalize_copy_text(str(item.get("task_type") or ""))
    code_markers = (
        "write_code",
        "write code",
        "code",
        "program",
        "python",
        "написать код",
        "напис",
        "программ",
    )
    return any(marker in task_type for marker in code_markers)


def render_practice_material_html(
    task: dict[str, Any],
    instances: dict[str, Any],
    *,
    html_template: HtmlFormatTemplate,
) -> str:
    lesson = task.get("lesson") if isinstance(task.get("lesson"), dict) else {}
    lesson_title = _practice_lesson_heading(lesson)
    lesson_goal = _plain_text(instances.get("lesson_goal"))
    lesson_objectives = _string_list(instances.get("lesson_objectives"))
    tasks = _list_of_dicts(instances.get("tasks"))
    intro_blocks = [
        f"<h1>{html.escape(str(lesson_title))}</h1>",
        "<p>Выполните задания в редакторе Python. Для каждой задачи используйте только условия, код и правила проверки, указанные в её блоке.</p>",
    ]
    if lesson_goal or lesson_objectives:
        goal_parts = ['<section id="goals"><h2 id="goals">Цели и задачи</h2>']
        if lesson_goal:
            goal_parts.append(f"<p><strong>Цель:</strong> {html.escape(lesson_goal)}</p>")
        if lesson_objectives:
            goal_parts.append("<ul>")
            for objective in lesson_objectives:
                if objective:
                    goal_parts.append(f"<li>{html.escape(objective)}</li>")
            goal_parts.append("</ul>")
        goal_parts.append("</section>")
        intro_blocks.append("".join(goal_parts))
    task_blocks: list[str] = []
    for index, item in enumerate(tasks, start=1):
        task_id = _first_nonempty(item.get("id"), f"P{index}")
        level = _plain_text(item.get("level"))
        scenario = _plain_text(item.get("scenario"))
        condition = _plain_text(item.get("student_condition"))
        code = _first_nonempty(item.get("faulty_code_display"), item.get("starter_code"))
        input_requirements = _plain_text(item.get("input_requirements"))
        output_requirements = _plain_text(item.get("output_requirements"))
        display_note = _plain_text(item.get("display_note"))
        tests = _list_of_dicts(item.get("runtime_tests")) or _list_of_dicts(item.get("tests"))
        manual_checks = item.get("manual_checks") if isinstance(item.get("manual_checks"), list) else []
        subtasks = item.get("subtasks") if isinstance(item.get("subtasks"), list) else []

        pieces = [f'<section id="{html.escape(task_id)}">', f'<h2 id="{html.escape(task_id)}">{html.escape(task_id)}</h2>']
        if level:
            pieces.append(f"<p><strong>Уровень:</strong> {html.escape(level)}</p>")
        if scenario:
            pieces.append(f"<p><strong>Ситуация:</strong> {html.escape(scenario)}</p>")
        if condition:
            pieces.append(f"<p><strong>Условие:</strong> {html.escape(condition)}</p>")
        if code:
            pieces.append(f"<p><strong>Код в редакторе:</strong></p><pre><code>{_html_code(code)}</code></pre>")
        elif _practice_task_needs_empty_code_block(item):
            pieces.append("<p><strong>Код в редакторе:</strong></p><pre><code></code></pre>")
        if display_note:
            pieces.append(f'<p class="cc-muted">{html.escape(display_note)}</p>')
        if input_requirements:
            pieces.append(f"<p><strong>Входные данные:</strong> {html.escape(input_requirements)}</p>")
        if output_requirements:
            pieces.append(f"<p><strong>Требование к результату:</strong> {html.escape(output_requirements)}</p>")
        pieces.append(_render_practice_subtasks(subtasks))
        pieces.append(_render_practice_tests(tests))
        pieces.append(_render_practice_manual_checks(manual_checks))
        if not tests and not manual_checks:
            pieces.append("<p><strong>Проверка:</strong> выполните запуск и сопоставьте результат с условием.</p>")
        pieces.append("</section>")
        task_blocks.append("".join(pieces))

    body_html = "".join(intro_blocks) + "".join(task_blocks)
    return html_template.render(body_html)


def render_current_control_material_html(
    task: dict[str, Any],
    autocheck: dict[str, Any],
    *,
    html_template: HtmlFormatTemplate,
) -> str:
    lesson = task.get("lesson") if isinstance(task.get("lesson"), dict) else {}
    lesson_number = _first_nonempty(lesson.get("lesson_number"), lesson.get("number"))
    lesson_title = _first_nonempty(lesson.get("title"), lesson.get("topic"), "Текущий контроль")
    heading = f"Занятие {lesson_number}. {lesson_title} — текущий контроль" if lesson_number else f"{lesson_title} — текущий контроль"
    question_blocks: list[str] = []
    for index, question in enumerate(_list_of_dicts(autocheck.get("questions")), start=1):
        question_id = _first_nonempty(question.get("id"), f"CC{index}")
        template_code = _plain_text(question.get("template_code"))
        question_type = _plain_text(question.get("question_type"))
        prompt = _plain_text(question.get("student_prompt"))
        options = _string_list(question.get("options"))
        expected_format = _plain_text(question.get("expected_answer_format"))
        correct_answers = _string_list(question.get("correct_answers"))
        visible_expected_format = _current_control_visible_expected_format(expected_format, correct_answers)
        config = question.get("autocheck_config") if isinstance(question.get("autocheck_config"), dict) else {}
        title_bits = [f"Вопрос {index} ({question_id})"]
        if template_code:
            title_bits.append(template_code)
        if question_type:
            title_bits.append(question_type)
        pieces = [
            f'<section id="{html.escape(question_id)}">',
            f"<h2>{html.escape(' — '.join(title_bits))}</h2>",
        ]
        if prompt:
            pieces.append(f"<p>{html.escape(prompt)}</p>")
        pieces.append(_render_current_control_question_body(question, options, config, visible_expected_format))
        pieces.append("</section>")
        question_blocks.append("".join(pieces))

    body_html = (
        f"<h1>{html.escape(str(heading))}</h1>"
        "<p>Выполните задания текущего контроля. Ответы и ключи проверяются во внутреннем слое платформы.</p>"
        + "".join(question_blocks)
    )
    return html_template.render(body_html)


def _render_current_control_question_body(
    question: dict[str, Any],
    options: list[str],
    config: dict[str, Any],
    expected_format: str,
) -> str:
    marker = _normalize_copy_text(
        " ".join(
            [
                str(question.get("template_code") or ""),
                str(question.get("question_type") or ""),
                str(config.get("type") or ""),
            ]
        )
    )
    if any(item in marker for item in ("matching", "8d", "соедин", "сопостав")):
        left_items = _string_list(config.get("left_items")) or options
        right_items = _string_list(config.get("right_items"))
        return _render_current_control_matching(left_items, right_items, expected_format)
    if any(item in marker for item in ("ordering", "6a", "order", "упорядоч")):
        display_items = _string_list(config.get("display_items")) or options
        return _render_current_control_options(display_items, ordered=True, expected_format=expected_format)
    if options:
        return _render_current_control_options(options, ordered=False, expected_format=expected_format)
    if expected_format:
        return f"<p><strong>Формат ответа:</strong> {html.escape(expected_format)}</p>"
    return "<p><strong>Формат ответа:</strong> введите ответ в поле платформы.</p>"


def _current_control_visible_expected_format(expected_format: str, correct_answers: list[str]) -> str:
    normalized_format = _normalize_copy_text(expected_format)
    if not normalized_format:
        return ""
    for answer in correct_answers:
        normalized_answer = _normalize_copy_text(answer)
        if normalized_answer and normalized_answer in normalized_format:
            return ""
    return expected_format


def _render_current_control_options(items: list[str], *, ordered: bool, expected_format: str) -> str:
    parts: list[str] = []
    if expected_format:
        parts.append(f"<p><strong>Формат ответа:</strong> {html.escape(expected_format)}</p>")
    tag = "ol" if ordered else "ul"
    parts.append(f"<{tag}>")
    for item in items:
        if item:
            parts.append(f"<li>{html.escape(item)}</li>")
    parts.append(f"</{tag}>")
    return "".join(parts)


def _render_current_control_matching(left_items: list[str], right_items: list[str], expected_format: str) -> str:
    parts = [
        "<p><strong>Формат ответа:</strong> соедините каждый пункт из списка A с одним пунктом из списка B.</p>"
    ]
    if expected_format:
        parts.append(f"<p>{html.escape(expected_format)}</p>")
    parts.append('<div class="cc-table-wrapper"><table class="cc-table"><thead><tr><th>Список A</th><th>Список B</th></tr></thead><tbody><tr><td><ol>')
    for item in left_items:
        if item:
            parts.append(f"<li>{html.escape(item)}</li>")
    parts.append("</ol></td><td><ol>")
    for item in right_items:
        if item:
            parts.append(f"<li>{html.escape(item)}</li>")
    parts.append("</ol></td></tr></tbody></table></div>")
    return "".join(parts)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _mentions_internal_practice_reference_field(value: Any) -> bool:
    text = str(value or "").lower()
    return "hidden_solution" in text or "teacher_explanation" in text


def _is_internal_practice_reference_field_path(value: Any) -> bool:
    path = str(value or "").lower()
    return bool(re.search(r"(^|[.\[\]])(hidden_solution|teacher_explanation)($|[.\[\]])", path))


def _filter_practice_internal_reference_field_issues(result: ValidationResult) -> ValidationResult:
    kept_block_issues: list[dict[str, Any]] = []
    removed_count = 0
    for issue in result.issues_by_block:
        if _is_internal_practice_reference_field_path(issue.get("field_path")):
            removed_count += 1
            continue
        kept_block_issues.append(issue)
    if removed_count == 0:
        return result

    issues = [issue for issue in result.issues if not _mentions_internal_practice_reference_field(issue)]
    fix_instructions = [
        instruction
        for instruction in result.fix_instructions
        if not _mentions_internal_practice_reference_field(instruction)
    ]
    blocking_block_issues = [
        issue
        for issue in kept_block_issues
        if str(issue.get("severity") or "blocking") != "non_blocking"
    ]
    return ValidationResult(
        approved=not issues and not blocking_block_issues,
        issues=issues,
        fix_instructions=fix_instructions,
        issues_by_block=kept_block_issues,
        passed_blocks=list(result.passed_blocks),
    )


def _controller_quality_score(data: dict[str, Any]) -> float:
    raw_score = data.get("quality_score", data.get("score"))
    if raw_score is None:
        return 5.0 if data.get("approved") else 0.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, score))


def _message_to_raw_response(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", type(message).__name__),
        "content": getattr(message, "content", ""),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        payload["invalid_tool_calls"] = invalid_tool_calls
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = additional_kwargs
    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = response_metadata
    return payload


def _raw_response_from_subagent_state(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return _message_to_raw_response(message)
        for message in reversed(messages):
            if isinstance(message, BaseMessage):
                return _message_to_raw_response(message)
    return {
        "state_keys": sorted(str(key) for key in state.keys()),
        "structured_response_type": type(state.get("structured_response")).__name__,
        "result_type": type(state.get("result")).__name__,
    }


def _raw_response_from_exception(exc: Exception) -> dict[str, Any]:
    for attr in ("raw_response", "response", "ai_message", "message"):
        value = getattr(exc, attr, None)
        if isinstance(value, BaseMessage):
            return _message_to_raw_response(value)
        if value is not None:
            return {attr: value}
    return {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }


def _subagent_langchain_config(
    base_config: Mapping[str, Any],
    *,
    thread_id: str,
    agent_type: str,
    schema: type[BaseModel],
) -> dict[str, Any]:
    config = dict(base_config or {})
    base_run_name = str(config.get("run_name") or "ismart_generator")
    config["run_name"] = f"{base_run_name}.{agent_type}"
    config["tags"] = list(dict.fromkeys([*(config.get("tags") or []), f"subagent:{agent_type}"]))
    config["metadata"] = {
        **(config.get("metadata") or {}),
        "subagent_type": agent_type,
        "structured_schema": schema.__name__,
    }
    configurable = dict(config.get("configurable") or {})
    configurable["thread_id"] = thread_id
    config["configurable"] = configurable
    return config


class StructuredSubagentCallState(TypedDict):
    agent_type: str
    system: str
    prompt: str
    schema: type[BaseModel]
    thread_id: str
    langchain_config: NotRequired[dict[str, Any]]
    structured_response: NotRequired[Any]
    raw_response: NotRequired[Any]


def _build_structured_subagent_call_graph(subagents: Mapping[str, Any]):
    builder = StateGraph(StructuredSubagentCallState)

    def route_node(state: StructuredSubagentCallState) -> dict[str, Any]:
        agent_type = state.get("agent_type")
        if agent_type not in subagents:
            raise KeyError(f"Subagent is not registered: {agent_type}")
        return {}

    def route_to_subagent(state: StructuredSubagentCallState) -> str:
        return str(state["agent_type"])

    builder.add_node("route_subagent", route_node)
    builder.add_edge(START, "route_subagent")
    builder.add_conditional_edges(
        "route_subagent",
        route_to_subagent,
        {agent_type: agent_type for agent_type in subagents},
    )
    for agent_type, subagent_graph in subagents.items():
        builder.add_node(agent_type, _make_structured_subagent_node(agent_type, subagent_graph))
        builder.add_edge(agent_type, END)
    return builder.compile(name="ismart_structured_subagent_call_graph")


def _make_structured_subagent_node(agent_type: str, subagent_graph: Any):
    def subagent_node(state: StructuredSubagentCallState) -> dict[str, Any]:
        schema = state["schema"]
        prompt = state["prompt"]
        child_state = subagent_graph.invoke(
            {
                "system_prompt": state["system"],
                "prompt": prompt,
                "messages": [HumanMessage(content=prompt)],
            },
            _subagent_langchain_config(
                state.get("langchain_config") or {},
                thread_id=state["thread_id"],
                agent_type=agent_type,
                schema=schema,
            ),
        )
        raw_response = _raw_response_from_subagent_state(child_state) if isinstance(child_state, dict) else None
        structured_response = None
        if isinstance(child_state, dict):
            structured_response = child_state.get("structured_response")
            if structured_response is None:
                structured_response = child_state.get("result")
        return {
            "structured_response": structured_response,
            "raw_response": raw_response,
        }

    return subagent_node


def _declares_multiple_valid_answers(*values: Any) -> bool:
    text = _normalize_copy_text("\n".join(str(value or "") for value in values))
    if not text:
        return False
    negative_markers = (
        "не несколько",
        "не допускает несколько",
        "not multiple",
        "not several",
    )
    if any(marker in text for marker in negative_markers):
        return False
    multiple_markers = (
        "любое из",
        "любой из",
        "любая из",
        "любые из",
        "каждое из",
        "каждый из",
        "каждая из",
        "несколько коррект",
        "несколько правиль",
        "несколько допустим",
        "более одного",
        "оба варианта",
        "оба ответа",
        "multiple valid",
        "more than one",
        "several correct",
        "any of",
    )
    return any(marker in text for marker in multiple_markers)


class StructuredSubagentInvoker:
    def __init__(
        self,
        subagents: Mapping[str, Any],
        *,
        trace: TraceLogger | None = None,
        langchain_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.subagents = subagents
        self.parent_graph = _build_structured_subagent_call_graph(subagents)
        self.trace = trace or TraceLogger()
        self.langchain_config = dict(langchain_config or {})

    def invoke(
        self,
        agent_type: str,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        if agent_type not in self.subagents:
            raise KeyError(f"Subagent is not registered: {agent_type}")
        raw_response: Any = None
        thread_id = f"ismart-{agent_type}-{schema.__name__}-{uuid4().hex}"
        self.trace.log(
            "subagent.invoke.start",
            agent_type=agent_type,
            schema=schema.__name__,
            thread_id=thread_id,
            prompt_chars=len(prompt),
        )
        try:
            state = self.parent_graph.invoke(
                {
                    "agent_type": agent_type,
                    "system": system,
                    "prompt": prompt,
                    "schema": schema,
                    "thread_id": thread_id,
                    "langchain_config": dict(self.langchain_config),
                },
                self.langchain_config,
            )
        except Exception as exc:
            raw_response = _raw_response_from_exception(exc)
            self.trace.log(
                "subagent.structured_output.exception",
                agent_type=agent_type,
                schema=schema.__name__,
                thread_id=thread_id,
                raw_response=raw_response,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise StructuredSubagentError(
                agent_type=agent_type,
                schema=schema,
                cause=exc,
                raw_response=raw_response,
            ) from exc
        if isinstance(state, dict):
            raw_response = state.get("raw_response") or _raw_response_from_subagent_state(state)
            result = state.get("structured_response")
            if result is None:
                result = state.get("result")
        else:
            result = None
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            try:
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(result)
                return schema.parse_obj(result)
            except Exception as exc:
                self.trace.log(
                    "subagent.structured_output.validation_error",
                    agent_type=agent_type,
                    schema=schema.__name__,
                    thread_id=thread_id,
                    raw_response=raw_response,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise StructuredSubagentError(
                    agent_type=agent_type,
                    schema=schema,
                    cause=exc,
                    raw_response=raw_response,
                ) from exc
        cause = TypeError(f"unsupported structured result: {type(result)!r}")
        self.trace.log(
            "subagent.structured_output.missing",
            agent_type=agent_type,
            schema=schema.__name__,
            thread_id=thread_id,
            raw_response=raw_response,
            result_type=type(result).__name__,
        )
        raise StructuredSubagentError(
            agent_type=agent_type,
            schema=schema,
            cause=cause,
            raw_response=raw_response,
        )


class StructuredSubagentError(RuntimeError):
    def __init__(
        self,
        *,
        agent_type: str,
        schema: type[BaseModel],
        cause: Exception,
        raw_response: Any = None,
    ) -> None:
        self.agent_type = agent_type
        self.schema = schema
        self.cause = cause
        self.raw_response = raw_response
        raw_suffix = f"; raw_response={raw_response!r}" if raw_response is not None else ""
        super().__init__(f"{agent_type} failed to return valid {schema.__name__}: {cause}{raw_suffix}")


class MaterialWorker:
    def __init__(
        self,
        *,
        subagents: Mapping[str, Any],
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
        trace: TraceLogger | None = None,
    ) -> None:
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()
        self.trace = trace or TraceLogger()
        self.invoker = StructuredSubagentInvoker(
            subagents,
            trace=self.trace,
            langchain_config=config.langchain_config,
        )

    def run(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
        initial_previous_issues: list[str] | None = None,
        attempts_dir: Path | None = None,
    ) -> MaterialResult:
        attempt_store = AttemptArtifactStore(attempts_dir)
        self.trace.log(
            "worker.start",
            kind=spec.kind,
            agent=spec.agent_type,
            dependency_kinds=list(spec.dependency_kinds),
            dependencies=[{"kind": item.kind, "status": item.status} for item in dependency_results],
        )
        blocked = [item for item in dependency_results if item.status != "approved"]
        if blocked:
            issues = [f"dependency {item.kind} has status {item.status}" for item in blocked]
            self.trace.log("worker.blocked_dependency", kind=spec.kind, issues=issues)
            return MaterialResult(
                kind=spec.kind,
                material_type=spec.material_type,
                agent_type=spec.agent_type,
                status="blocked_dependency",
                iterations=0,
                content="",
                prompt_files=spec.prompt_files,
                validation_issues=issues,
            )

        prompt_contents = read_prompt_files(self.config, spec.prompt_files)
        self.trace.log("worker.prompt_files_loaded", kind=spec.kind, prompt_files=list(spec.prompt_files))
        html_format_template = load_html_format_template() if spec.kind in {"practice", "current_control"} else None
        if html_format_template is not None:
            self.trace.log(
                "worker.html_template_loaded",
                kind=spec.kind,
                source=str(HTML_TEMPLATE_PATH),
                style_chars=len(html_format_template.style_block),
            )
        previous_content = ""
        previous_issues = list(initial_previous_issues or [])
        previous_validation: ValidationResult | None = None
        last_agent_notes: list[str] = []
        last_rule_result: ValidationResult | None = None
        last_llm_result: ValidationResult | None = None
        last_validation: ValidationResult | None = None
        previous_artifacts: dict[str, Any] = {}
        last_generation_artifacts: dict[str, Any] = {}
        if previous_issues:
            self.trace.log("worker.initial_issues", kind=spec.kind, issues=previous_issues)

        for attempt in range(1, self.config.max_generation_iterations + 1):
            self.trace.log(
                "worker.attempt.start",
                kind=spec.kind,
                attempt=attempt,
                max_attempts=self.config.max_generation_iterations,
                previous_content_chars=len(previous_content),
                previous_issues_count=len(previous_issues),
            )
            if spec.kind == "practice":
                generated_attempt = self._generate_practice_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                    html_template=html_format_template,
                )
            elif spec.kind == "self_work":
                generated_attempt = self._generate_self_work_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                )
            elif spec.kind == "current_control":
                generated_attempt = self._generate_current_control_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                    html_template=html_format_template,
                )
            elif spec.kind == "intermediate":
                generated_attempt = self._generate_intermediate_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                )
            else:
                generated_attempt = self._generate_material_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                )
            raw_content = generated_attempt.raw_content
            content = generated_attempt.content
            agent_notes = generated_attempt.agent_notes
            generation_artifacts = generated_attempt.generation_artifacts
            last_generation_artifacts = generation_artifacts
            self.trace.log(
                "worker.generator.done",
                kind=spec.kind,
                attempt=attempt,
                content_chars=len(content),
                raw_content_chars=len(raw_content),
                agent_notes_count=len(agent_notes),
            )
            self.trace.log(
                "worker.content_boundary",
                kind=spec.kind,
                attempt=attempt,
                starts_with=content[:80],
                ends_with=content[-80:] if content else "",
                boundary_issues=generated_attempt.boundary_issues,
            )
            rule_result: ValidationResult | None = None
            llm_result: ValidationResult | None = None
            if not content:
                validation = generated_attempt.structural_validation.merge(
                    ValidationResult.fail([f"{spec.agent_type} returned empty content"])
                )
                self.trace.log("worker.validation.empty_content", kind=spec.kind, attempt=attempt)
            else:
                rule_result = self.rule_validator.validate_material(content, spec, task)
                if generated_attempt.boundary_issues:
                    rule_result = ValidationResult.fail(generated_attempt.boundary_issues).merge(rule_result)
                rule_result = generated_attempt.structural_validation.merge(rule_result)
                self.trace.log(
                    "worker.rule_validation.done",
                    kind=spec.kind,
                    attempt=attempt,
                    approved=rule_result.approved,
                    issues=rule_result.issues,
                )
                llm_result = self._validate_with_llm(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    content=content,
                    rule_result=rule_result,
                    generation_artifacts=generation_artifacts,
                )
                validation = rule_result.merge(llm_result)
                self.trace.log(
                    "worker.validation.merged",
                    kind=spec.kind,
                    attempt=attempt,
                    approved=validation.approved,
                    issues=validation.issues,
                )
            last_agent_notes = agent_notes
            last_rule_result = rule_result
            last_llm_result = llm_result
            last_validation = validation
            attempt_store.write_material_attempt(
                kind=spec.kind,
                attempt=attempt,
                raw_content=raw_content,
                content=content,
                rule_result=rule_result,
                llm_result=llm_result,
                validation=validation,
                boundary_issues=generated_attempt.boundary_issues,
                agent_notes=agent_notes,
                metadata={
                    "agent_type": spec.agent_type,
                    "material_type": spec.material_type,
                    "raw_content_chars": len(raw_content),
                    "content_chars": len(content),
                    "generation_artifacts": generation_artifacts,
                },
            )

            if validation.approved:
                self.trace.log("worker.approved", kind=spec.kind, attempt=attempt, content_chars=len(content))
                return MaterialResult(
                    kind=spec.kind,
                    material_type=spec.material_type,
                    agent_type=spec.agent_type,
                    status="approved",
                    iterations=attempt,
                    content=content,
                    prompt_files=spec.prompt_files,
                    validation_issues_by_block=validation.issues_by_block,
                    validation_passed_blocks=validation.passed_blocks,
                    agent_notes=agent_notes,
                    generation_artifacts=generation_artifacts,
                )

            previous_content = content
            previous_issues = validation.issues
            previous_validation = validation
            previous_artifacts = generation_artifacts
            if attempt < self.config.max_generation_iterations:
                self.trace.log("worker.retry", kind=spec.kind, next_attempt=attempt + 1, issues=previous_issues)

        controller_decision = self._review_validation_failure(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependency_results=dependency_results,
            content=previous_content,
            rule_result=last_rule_result,
            llm_result=last_llm_result,
            validation=last_validation,
            generation_artifacts=last_generation_artifacts,
            attempt_store=attempt_store,
        )
        controller_score = float(controller_decision.get("quality_score", 0.0) or 0.0)
        if controller_score >= self.config.validation_controller_accept_score:
            rationale = str(controller_decision.get("rationale") or "validator rejection was not blocking")
            self.trace.log(
                "worker.controller.accepted_by_score",
                kind=spec.kind,
                quality_score=controller_score,
                accept_score=self.config.validation_controller_accept_score,
                controller_approved=controller_decision.get("approved"),
                rationale=rationale,
            )
            return MaterialResult(
                kind=spec.kind,
                material_type=spec.material_type,
                agent_type=spec.agent_type,
                status="approved",
                iterations=self.config.max_generation_iterations,
                content=previous_content,
                prompt_files=spec.prompt_files,
                validation_issues=[],
                validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
                validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
                agent_notes=[
                    *last_agent_notes,
                    (
                        "ValidationControllerAgent accepted after validator review "
                        f"with quality_score={controller_score:g}: {rationale}"
                    ),
                ],
                controller_called=True,
                controller_decision=controller_decision,
                generation_artifacts=last_generation_artifacts,
            )

        if controller_decision:
            previous_issues = [str(item) for item in controller_decision.get("blocking_issues") or previous_issues]
            self.trace.log("worker.controller.kept_failed", kind=spec.kind, issues=previous_issues)

        self.trace.log("worker.failed", kind=spec.kind, issues=previous_issues)
        return MaterialResult(
            kind=spec.kind,
            material_type=spec.material_type,
            agent_type=spec.agent_type,
            status="failed",
            iterations=self.config.max_generation_iterations,
            content=previous_content,
            prompt_files=spec.prompt_files,
            validation_issues=previous_issues,
            validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
            validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
            controller_called=bool(controller_decision),
            controller_decision=controller_decision,
            generation_artifacts=last_generation_artifacts,
        )

    def _generate_material_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
    ) -> GeneratedAttempt:
        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[str(item) for item in generated_model.agent_notes],
            generation_artifacts={},
            structural_validation=ValidationResult.ok(),
        )

    def _generate_practice_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
        html_template: HtmlFormatTemplate | None,
    ) -> GeneratedAttempt:
        if html_template is None:
            raise ValueError("Practice HTML template was not loaded.")
        templates = copy.deepcopy(previous_artifacts.get("practice_templates"))
        instances = copy.deepcopy(previous_artifacts.get("practice_instances"))
        reuse_templates = False
        reuse_instances = False
        if isinstance(templates, dict):
            template_validation = self._validate_practice_templates(task=task, spec=spec, templates=templates)
            reuse_templates = template_validation.approved
            if reuse_templates:
                self.trace.log("worker.practice_templates.reused", attempt=attempt)
            else:
                templates = {}
        else:
            templates = {}

        if (
            reuse_templates
            and isinstance(instances, dict)
            and previous_validation is not None
            and previous_validation.approved
        ):
            instance_validation = self._validate_practice_instances(
                task=task,
                spec=spec,
                templates=templates,
                instances=instances,
            )
            reuse_instances = instance_validation.approved
            if reuse_instances:
                self.trace.log("worker.practice_artifacts.reused", attempt=attempt)
                artifacts = {
                    "practice_templates": templates,
                    "practice_instances": instances,
                }
                attempt_store.write_practice_generation_artifacts(
                    attempt=attempt,
                    templates=templates,
                    instances=instances,
                    metadata={"stage": "instances_frozen"},
                )
        else:
            instances = {}

        if not reuse_instances:
            if not reuse_templates:
                template_prompt = build_practice_template_prompt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependencies=dependency_results,
                    previous_artifacts=previous_artifacts,
                    previous_issues=previous_issues,
                )
                template_model = self.invoker.invoke(
                    "PracticeTaskTemplateAgent",
                    system=build_practice_template_system_prompt(),
                    prompt=template_prompt,
                    schema=PracticeTaskTemplateSet,
                )
                if not isinstance(template_model, PracticeTaskTemplateSet):
                    raise TypeError(
                        f"PracticeTaskTemplateAgent returned {type(template_model)!r}, expected PracticeTaskTemplateSet"
                    )
                templates = _model_to_dict(template_model)
                template_validation = self._validate_practice_templates(task=task, spec=spec, templates=templates)
                self.trace.log(
                    "worker.practice_templates.done",
                    attempt=attempt,
                    approved=template_validation.approved,
                    issues=template_validation.issues,
                )

                if not template_validation.approved:
                    artifacts = {
                        "practice_templates": templates,
                        "practice_instances": {},
                    }
                    attempt_store.write_practice_generation_artifacts(
                        attempt=attempt,
                        templates=templates,
                        instances=None,
                        metadata={"stage": "templates"},
                    )
                    return GeneratedAttempt(
                        raw_content="",
                        content="",
                        boundary_issues=[],
                        agent_notes=[*templates.get("agent_notes", []), "Practice template structural validation failed."],
                        generation_artifacts=artifacts,
                        structural_validation=template_validation,
                    )

            variant_prompt = build_practice_variant_prompt(
                task=task,
                spec=spec,
                prompt_contents=prompt_contents,
                references=references,
                dependencies=dependency_results,
                templates=templates,
                previous_artifacts=previous_artifacts,
                previous_issues=previous_issues,
                previous_validation=previous_validation,
            )
            instance_model = self.invoker.invoke(
                "PracticeTaskVariantAgent",
                system=build_practice_variant_system_prompt(),
                prompt=variant_prompt,
                schema=PracticeTaskInstanceSet,
            )
            if not isinstance(instance_model, PracticeTaskInstanceSet):
                raise TypeError(
                    f"PracticeTaskVariantAgent returned {type(instance_model)!r}, expected PracticeTaskInstanceSet"
                )
            instances = _model_to_dict(instance_model)
            instances = _normalize_practice_instance_tests(instances)
            instance_validation = self._validate_practice_instances(
                task=task,
                spec=spec,
                templates=templates,
                instances=instances,
            )
            self.trace.log(
                "worker.practice_instances.done",
                attempt=attempt,
                approved=instance_validation.approved,
                issues=instance_validation.issues,
            )

            artifacts = {
                "practice_templates": templates,
                "practice_instances": instances,
            }
            attempt_store.write_practice_generation_artifacts(
                attempt=attempt,
                templates=templates,
                instances=instances,
                metadata={"stage": "instances"},
            )
            if not instance_validation.approved:
                return GeneratedAttempt(
                    raw_content="",
                    content="",
                    boundary_issues=[],
                    agent_notes=[
                        *templates.get("agent_notes", []),
                        *instances.get("agent_notes", []),
                        "Practice instance structural validation failed.",
                    ],
                    generation_artifacts=artifacts,
                    structural_validation=instance_validation,
                )

        raw_content = render_practice_material_html(
            task,
            instances,
            html_template=html_template,
        )
        boundary = isolate_material_html(raw_content)
        self.trace.log(
            "worker.practice_renderer.done",
            attempt=attempt,
            content_chars=len(boundary.content),
        )
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *templates.get("agent_notes", []),
                *instances.get("agent_notes", []),
                "Practice HTML rendered deterministically from practice_instances.",
            ],
            generation_artifacts=artifacts,
            structural_validation=ValidationResult.ok(),
        )

    def _generate_self_work_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
    ) -> GeneratedAttempt:
        autocheck = copy.deepcopy(previous_artifacts.get("self_work_autocheck"))
        artifact_validation = (
            self._validate_self_work_autocheck(autocheck) if isinstance(autocheck, dict) else ValidationResult.fail([])
        )
        frozen_artifacts = isinstance(autocheck, dict) and artifact_validation.approved
        if not frozen_artifacts:
            autocheck_prompt = build_self_work_autocheck_prompt(
                task=task,
                spec=spec,
                prompt_contents=prompt_contents,
                references=references,
                dependencies=dependency_results,
                previous_artifacts=previous_artifacts,
                previous_issues=previous_issues,
            )
            autocheck_model = self.invoker.invoke(
                "SelfWorkAutocheckAgent",
                system=build_self_work_autocheck_system_prompt(),
                prompt=autocheck_prompt,
                schema=SelfWorkAutocheckSet,
            )
            if not isinstance(autocheck_model, SelfWorkAutocheckSet):
                raise TypeError(
                    f"SelfWorkAutocheckAgent returned {type(autocheck_model)!r}, expected SelfWorkAutocheckSet"
                )
            autocheck = _model_to_dict(autocheck_model)
            artifact_validation = self._validate_self_work_autocheck(autocheck)
        artifacts = {
            "self_work_autocheck": autocheck,
            "self_work_autocheck_check": {
                "approved": artifact_validation.approved,
                "issues": artifact_validation.issues,
            },
        }
        attempt_store.write_self_work_generation_artifacts(
            attempt=attempt,
            autocheck=autocheck,
            structural_check=artifacts["self_work_autocheck_check"],
            metadata={"stage": "autocheck_frozen" if frozen_artifacts and artifact_validation.approved else "autocheck"},
        )
        if frozen_artifacts and artifact_validation.approved:
            self.trace.log("worker.self_work_autocheck.reused", attempt=attempt)
        self.trace.log(
            "worker.self_work_autocheck.done",
            attempt=attempt,
            approved=artifact_validation.approved,
            issues=artifact_validation.issues,
        )

        if not artifact_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *autocheck.get("agent_notes", []),
                    "Self-work autocheck artifact structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=artifact_validation,
            )

        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
            generation_artifacts=artifacts,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *autocheck.get("agent_notes", []),
                *[str(item) for item in generated_model.agent_notes],
            ],
            generation_artifacts=artifacts,
            structural_validation=ValidationResult.ok(),
        )

    def _validate_self_work_autocheck(self, autocheck: dict[str, Any]) -> ValidationResult:
        issues: list[str] = []
        independent_tasks = autocheck.get("independent_tasks") if isinstance(autocheck, dict) else None
        selfcheck_questions = autocheck.get("selfcheck_questions") if isinstance(autocheck, dict) else None
        if not isinstance(independent_tasks, list):
            issues.append("self_work_autocheck.independent_tasks must be a list")
            independent_tasks = []
        if not isinstance(selfcheck_questions, list):
            issues.append("self_work_autocheck.selfcheck_questions must be a list")
            selfcheck_questions = []

        if len(independent_tasks) != 8:
            issues.append(f"self_work_autocheck independent_tasks count mismatch: expected 8, got {len(independent_tasks)}")
        if len(selfcheck_questions) != 10:
            issues.append(
                f"self_work_autocheck selfcheck_questions count mismatch: expected 10, got {len(selfcheck_questions)}"
            )

        task_ids: list[str] = []
        for index, item in enumerate(independent_tasks, start=1):
            if not isinstance(item, dict):
                issues.append(f"self_work_autocheck.independent_tasks[{index}] must be an object")
                continue
            task_id = str(item.get("id") or f"?{index}")
            task_ids.append(task_id)
            for field in ("id", "student_task_title", "checked_skill", "checking_mode"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"self_work_autocheck.independent_tasks.{task_id} missing required field {field}")
            runtime_tests = item.get("runtime_tests")
            manual_rules = item.get("manual_check_rules")
            has_check = bool(str(item.get("correct_answer") or "").strip())
            has_check = has_check or (isinstance(runtime_tests, list) and bool(runtime_tests))
            has_check = has_check or (isinstance(manual_rules, list) and bool(manual_rules))
            if not has_check:
                issues.append(
                    f"self_work_autocheck.independent_tasks.{task_id} needs correct_answer, runtime_tests, or manual_check_rules"
                )
            if not isinstance(runtime_tests, list):
                issues.append(f"self_work_autocheck.independent_tasks.{task_id} runtime_tests must be a list")
            if not isinstance(manual_rules, list):
                issues.append(f"self_work_autocheck.independent_tasks.{task_id} manual_check_rules must be a list")

        question_ids: list[str] = []
        for index, item in enumerate(selfcheck_questions, start=1):
            if not isinstance(item, dict):
                issues.append(f"self_work_autocheck.selfcheck_questions[{index}] must be an object")
                continue
            question_id = str(item.get("id") or f"?{index}")
            question_ids.append(question_id)
            for field in ("id", "template_code", "question_type", "skill_target", "student_prompt"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} missing required field {field}")
            correct_answers = item.get("correct_answers")
            if not isinstance(correct_answers, list) or not any(str(answer).strip() for answer in correct_answers):
                issues.append(
                    f"self_work_autocheck.selfcheck_questions.{question_id} needs at least one correct answer"
                )
            if not isinstance(item.get("options"), list):
                issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} options must be a list")
            if not isinstance(item.get("autocheck_config"), dict):
                issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} autocheck_config must be an object")

        if len(task_ids) != len(set(task_ids)):
            issues.append("self_work_autocheck.independent_tasks ids must be unique")
        if len(question_ids) != len(set(question_ids)):
            issues.append("self_work_autocheck.selfcheck_questions ids must be unique")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _generate_current_control_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
        html_template: HtmlFormatTemplate | None,
    ) -> GeneratedAttempt:
        if html_template is None:
            raise ValueError("Current-control HTML template was not loaded.")
        autocheck = copy.deepcopy(previous_artifacts.get("current_control_autocheck"))
        artifact_validation = (
            self._validate_current_control_autocheck(autocheck)
            if isinstance(autocheck, dict)
            else ValidationResult.fail([])
        )
        frozen_artifacts = isinstance(autocheck, dict) and artifact_validation.approved
        if not frozen_artifacts:
            autocheck_prompt = build_current_control_autocheck_prompt(
                task=task,
                spec=spec,
                prompt_contents=prompt_contents,
                references=references,
                dependencies=dependency_results,
                previous_artifacts=previous_artifacts,
                previous_issues=previous_issues,
            )
            try:
                autocheck_model = self.invoker.invoke(
                    "CurrentControlAutocheckAgent",
                    system=build_current_control_autocheck_system_prompt(),
                    prompt=autocheck_prompt,
                    schema=CurrentControlAutocheckSet,
                )
            except StructuredSubagentError as exc:
                artifact_validation = ValidationResult.fail([str(exc)])
                artifacts = {
                    "current_control_autocheck": {},
                    "current_control_autocheck_check": {
                        "approved": False,
                        "issues": artifact_validation.issues,
                    },
                }
                attempt_store.write_current_control_generation_artifacts(
                    attempt=attempt,
                    autocheck=artifacts["current_control_autocheck"],
                    structural_check=artifacts["current_control_autocheck_check"],
                    metadata={"stage": "autocheck", "structured_output_error": str(exc)},
                )
                self.trace.log(
                    "worker.current_control_autocheck.structured_output_failed",
                    attempt=attempt,
                    issues=artifact_validation.issues,
                )
                return GeneratedAttempt(
                    raw_content="",
                    content="",
                    boundary_issues=[],
                    agent_notes=[str(exc)],
                    generation_artifacts=artifacts,
                    structural_validation=artifact_validation,
                )
            if not isinstance(autocheck_model, CurrentControlAutocheckSet):
                raise TypeError(
                    "CurrentControlAutocheckAgent returned "
                    f"{type(autocheck_model)!r}, expected CurrentControlAutocheckSet"
                )

            autocheck = _model_to_dict(autocheck_model)
            artifact_validation = self._validate_current_control_autocheck(autocheck)
        if frozen_artifacts:
            self.trace.log("worker.current_control_autocheck.reused", attempt=attempt)
        artifacts = {
            "current_control_autocheck": autocheck,
            "current_control_autocheck_check": {
                "approved": artifact_validation.approved,
                "issues": artifact_validation.issues,
            },
        }
        attempt_store.write_current_control_generation_artifacts(
            attempt=attempt,
            autocheck=autocheck,
            structural_check=artifacts["current_control_autocheck_check"],
            metadata={"stage": "autocheck_frozen" if frozen_artifacts else "autocheck"},
        )
        self.trace.log(
            "worker.current_control_autocheck.done",
            attempt=attempt,
            approved=artifact_validation.approved,
            issues=artifact_validation.issues,
        )

        if not artifact_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *autocheck.get("agent_notes", []),
                    "Current-control autocheck artifact structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=artifact_validation,
            )

        raw_content = render_current_control_material_html(
            task,
            autocheck,
            html_template=html_template,
        )
        boundary = isolate_material_html(raw_content)
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *autocheck.get("agent_notes", []),
                "Current-control HTML rendered deterministically from current_control_autocheck.",
            ],
            generation_artifacts=artifacts,
            structural_validation=ValidationResult.ok(),
        )

    def _validate_current_control_autocheck(self, autocheck: dict[str, Any]) -> ValidationResult:
        issues: list[str] = []
        questions = autocheck.get("questions") if isinstance(autocheck, dict) else None
        if not isinstance(questions, list):
            issues.append("current_control_autocheck.questions must be a list")
            questions = []

        if len(questions) != 3:
            issues.append(f"current_control_autocheck question count mismatch: expected 3, got {len(questions)}")

        question_ids: list[str] = []
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                issues.append(f"current_control_autocheck.questions[{index}] must be an object")
                continue
            question_id = str(item.get("id") or f"?{index}")
            question_ids.append(question_id)
            for field in ("id", "template_code", "question_type", "skill_target", "student_prompt"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"current_control_autocheck.questions.{question_id} missing required field {field}")

            correct_answers = item.get("correct_answers")
            if not isinstance(correct_answers, list) or not any(str(answer).strip() for answer in correct_answers):
                issues.append(
                    f"current_control_autocheck.questions.{question_id} needs at least one correct answer"
                )
            if not isinstance(item.get("options"), list):
                issues.append(f"current_control_autocheck.questions.{question_id} options must be a list")
            autocheck_config = item.get("autocheck_config")
            if not isinstance(autocheck_config, dict) or not autocheck_config:
                issues.append(
                    f"current_control_autocheck.questions.{question_id} needs non-empty autocheck_config"
                )

            question_type = _normalize_copy_text(str(item.get("question_type") or ""))
            template_code = _normalize_copy_text(str(item.get("template_code") or ""))
            is_open_answer = any(
                marker in f"{question_type} {template_code}"
                for marker in ("open", "text", "input", "3h", "free")
            )
            if is_open_answer and not str(item.get("expected_answer_format") or "").strip():
                issues.append(
                    f"current_control_autocheck.questions.{question_id} open-answer item needs expected_answer_format"
                )

        if len(question_ids) != len(set(question_ids)):
            issues.append("current_control_autocheck.questions ids must be unique")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _generate_intermediate_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
    ) -> GeneratedAttempt:
        assessment = copy.deepcopy(previous_artifacts.get("intermediate_assessment"))
        artifact_validation = (
            self._validate_intermediate_assessment_artifact(assessment)
            if isinstance(assessment, dict)
            else ValidationResult.fail([])
        )
        frozen_artifacts = isinstance(assessment, dict) and artifact_validation.approved
        if not frozen_artifacts:
            artifact_prompt = build_intermediate_assessment_artifact_prompt(
                task=task,
                spec=spec,
                prompt_contents=prompt_contents,
                references=references,
                dependencies=dependency_results,
                previous_artifacts=previous_artifacts,
                previous_issues=previous_issues,
            )
            artifact_model = self.invoker.invoke(
                "IntermediateAssessmentArtifactAgent",
                system=build_intermediate_assessment_artifact_system_prompt(),
                prompt=artifact_prompt,
                schema=IntermediateAssessmentArtifact,
            )
            if not isinstance(artifact_model, IntermediateAssessmentArtifact):
                raise TypeError(
                    f"IntermediateAssessmentArtifactAgent returned {type(artifact_model)!r}, "
                    "expected IntermediateAssessmentArtifact"
                )

            assessment = _model_to_dict(artifact_model)
            assessment = self._normalize_intermediate_assessment_display_order(assessment)
            artifact_validation = self._validate_intermediate_assessment_artifact(assessment)
        else:
            self.trace.log("worker.intermediate_assessment.reused", attempt=attempt)
        artifacts = {
            "intermediate_assessment": assessment,
            "intermediate_assessment_check": {
                "approved": artifact_validation.approved,
                "issues": artifact_validation.issues,
            },
        }
        attempt_store.write_intermediate_assessment_artifacts(
            attempt=attempt,
            artifact=assessment,
            structural_check=artifacts["intermediate_assessment_check"],
            metadata={"stage": "assessment_artifact_frozen" if frozen_artifacts else "assessment_artifact"},
        )
        self.trace.log(
            "worker.intermediate_assessment.done",
            attempt=attempt,
            approved=artifact_validation.approved,
            issues=artifact_validation.issues,
        )

        if not artifact_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *assessment.get("agent_notes", []),
                    "Intermediate assessment artifact structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=artifact_validation,
            )

        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
            generation_artifacts=artifacts,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *assessment.get("agent_notes", []),
                *[str(item) for item in generated_model.agent_notes],
            ],
            generation_artifacts=artifacts,
            structural_validation=ValidationResult.ok(),
        )

    def _normalize_intermediate_assessment_display_order(self, assessment: dict[str, Any]) -> dict[str, Any]:
        variants = assessment.get("variants") if isinstance(assessment, dict) else None
        if not isinstance(variants, list):
            return assessment

        notes: list[str] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            variant_id = str(variant.get("id") or "?")
            for question in variant.get("test_questions") or []:
                if not isinstance(question, dict):
                    continue
                question_id = str(question.get("id") or "?")
                template_code = str(question.get("template_code") or "").strip().upper()
                if template_code == "6A":
                    if self._normalize_intermediate_ordering_question_display(question):
                        notes.append(f"{variant_id}.{question_id}: normalized 6A display order.")
                if template_code in {"6G", "8D"} or self._intermediate_pair_map(question):
                    if self._normalize_intermediate_pairing_question_display(question):
                        notes.append(f"{variant_id}.{question_id}: normalized matching right_items display order.")

        if notes:
            agent_notes = assessment.get("agent_notes")
            if not isinstance(agent_notes, list):
                agent_notes = []
            assessment["agent_notes"] = [*agent_notes, *notes]
        return assessment

    def _normalize_intermediate_ordering_question_display(self, question: dict[str, Any]) -> bool:
        correct_order = self._intermediate_ordering_correct_items(question)
        if len(correct_order) < 2:
            return False
        options = question.get("options")
        display_items = [str(item) for item in options] if isinstance(options, list) and options else list(correct_order)
        if len(display_items) != len(correct_order):
            display_items = list(correct_order)

        deranged = self._derange_items_against_positions(display_items, correct_order)
        if deranged == display_items:
            return False

        question["options"] = deranged
        autocheck = question.get("autocheck_config")
        if isinstance(autocheck, dict):
            autocheck["display_items"] = deranged
            if isinstance(autocheck.get("items"), list):
                autocheck["items"] = deranged
        return True

    def _normalize_intermediate_pairing_question_display(self, question: dict[str, Any]) -> bool:
        autocheck = question.get("autocheck_config")
        if not isinstance(autocheck, dict):
            return False
        left_items = self._intermediate_autocheck_items(autocheck, "left_items", "left")
        right_items = self._intermediate_autocheck_items(autocheck, "right_items", "right")
        pair_map = self._intermediate_pair_map(question)
        if len(left_items) < 2 or len(right_items) < 2 or not pair_map:
            return False

        correct_by_position = [pair_map.get(_normalize_copy_text(left), "") for left in left_items]
        deranged = self._derange_items_against_positions(right_items, correct_by_position)
        if deranged == right_items:
            return False

        autocheck["right_items"] = deranged
        options = question.get("options")
        if isinstance(options, list):
            left_norms = {_normalize_copy_text(item) for item in left_items}
            right_norms = {_normalize_copy_text(item) for item in right_items}
            remaining = [
                str(item)
                for item in options
                if _normalize_copy_text(str(item)) not in left_norms
                and _normalize_copy_text(str(item)) not in right_norms
            ]
            question["options"] = [*left_items, *deranged, *remaining]
        return True

    def _intermediate_ordering_correct_items(self, question: dict[str, Any]) -> list[str]:
        autocheck = question.get("autocheck_config")
        if isinstance(autocheck, dict):
            for key in ("ordered_items", "items_in_correct_order", "correct_order"):
                items = self._string_items(autocheck.get(key))
                if items:
                    return items
        return self._string_items(question.get("correct_answers"))

    @staticmethod
    def _intermediate_autocheck_items(autocheck: dict[str, Any], *keys: str) -> list[str]:
        for key in keys:
            items = MaterialWorker._string_items(autocheck.get(key))
            if items:
                return items
        return []

    @staticmethod
    def _string_items(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("label") or item.get("value") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                items.append(text)
        return items

    def _intermediate_pair_map(self, question: dict[str, Any]) -> dict[str, str]:
        pairs: dict[str, str] = {}
        autocheck = question.get("autocheck_config")
        if isinstance(autocheck, dict):
            raw_pairs = autocheck.get("correct_pairs")
            if isinstance(raw_pairs, dict):
                for left, right in raw_pairs.items():
                    if str(left).strip() and str(right).strip():
                        pairs[_normalize_copy_text(str(left))] = _normalize_copy_text(str(right))
            elif isinstance(raw_pairs, list):
                for pair in raw_pairs:
                    if isinstance(pair, list) and len(pair) >= 2:
                        left = str(pair[0] or "").strip()
                        right = str(pair[1] or "").strip()
                        if left and right:
                            pairs[_normalize_copy_text(left)] = _normalize_copy_text(right)

        for left, right in self._intermediate_matching_pairs(question):
            if left and right:
                pairs[_normalize_copy_text(left)] = _normalize_copy_text(right)
        return pairs

    @staticmethod
    def _derange_items_against_positions(items: list[str], forbidden_by_position: list[str]) -> list[str]:
        if len(items) < 2:
            return items
        normalized_forbidden = [_normalize_copy_text(item) for item in forbidden_by_position]

        def valid(candidate: list[str]) -> bool:
            return all(
                index >= len(normalized_forbidden)
                or not normalized_forbidden[index]
                or _normalize_copy_text(item) != normalized_forbidden[index]
                for index, item in enumerate(candidate)
            )

        rotated = items[1:] + items[:1]
        if valid(rotated):
            return rotated
        reversed_items = list(reversed(items))
        if valid(reversed_items):
            return reversed_items

        def search(prefix: list[str], remaining: list[str]) -> list[str] | None:
            index = len(prefix)
            if not remaining:
                return prefix if valid(prefix) else None
            for position, item in enumerate(remaining):
                if index < len(normalized_forbidden) and _normalize_copy_text(item) == normalized_forbidden[index]:
                    continue
                result = search([*prefix, item], [*remaining[:position], *remaining[position + 1 :]])
                if result is not None:
                    return result
            return None

        return search([], items) or items

    def _validate_intermediate_assessment_artifact(self, assessment: dict[str, Any]) -> ValidationResult:
        issues: list[str] = []
        variants = assessment.get("variants") if isinstance(assessment, dict) else None
        if not isinstance(variants, list):
            return ValidationResult.fail(["intermediate_assessment.variants must be a list"])
        if len(variants) != 4:
            issues.append(f"intermediate_assessment variants count mismatch: expected 4, got {len(variants)}")

        variant_ids: list[str] = []
        for variant_index, variant in enumerate(variants, start=1):
            if not isinstance(variant, dict):
                issues.append(f"intermediate_assessment.variants[{variant_index}] must be an object")
                continue
            variant_id = str(variant.get("id") or f"?{variant_index}")
            variant_ids.append(variant_id)
            for field in ("id", "title"):
                if not str(variant.get(field) or "").strip():
                    issues.append(f"intermediate_assessment.{variant_id} missing required field {field}")

            test_questions = variant.get("test_questions")
            open_code_questions = variant.get("open_code_questions")
            code_tasks = variant.get("code_tasks")
            if not isinstance(test_questions, list):
                issues.append(f"intermediate_assessment.{variant_id}.test_questions must be a list")
                test_questions = []
            if not isinstance(open_code_questions, list):
                issues.append(f"intermediate_assessment.{variant_id}.open_code_questions must be a list")
                open_code_questions = []
            if not isinstance(code_tasks, list):
                issues.append(f"intermediate_assessment.{variant_id}.code_tasks must be a list")
                code_tasks = []

            if len(test_questions) != 5:
                issues.append(
                    f"intermediate_assessment.{variant_id} test_questions count mismatch: expected 5, got {len(test_questions)}"
                )
            if len(open_code_questions) != 5:
                issues.append(
                    f"intermediate_assessment.{variant_id} open_code_questions count mismatch: expected 5, got {len(open_code_questions)}"
                )
            if len(code_tasks) != 5:
                issues.append(
                    f"intermediate_assessment.{variant_id} code_tasks count mismatch: expected 5, got {len(code_tasks)}"
                )

            item_ids: list[str] = []
            for item in test_questions:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.test_questions contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "template_code", "skill_target", "student_prompt"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.test_questions.{item_id} missing {field}")
                answers = item.get("correct_answers")
                if not isinstance(answers, list) or not any(str(answer).strip() for answer in answers):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.test_questions.{item_id} needs at least one correct answer"
                    )
                else:
                    answer_values = [str(answer).strip() for answer in answers if str(answer).strip()]
                    if len(answer_values) == 1 and _declares_multiple_valid_answers(
                        item.get("student_prompt"),
                        item.get("internal_explanation"),
                        item.get("autocheck_config"),
                    ):
                        issues.append(
                            f"intermediate_assessment.{variant_id}.test_questions.{item_id} declares multiple "
                            "valid answers but provides exactly one correct answer; make the criterion unique or "
                            "include every correct answer with a compatible template/autocheck_config"
                        )
                if not isinstance(item.get("options"), list):
                    issues.append(f"intermediate_assessment.{variant_id}.test_questions.{item_id} options must be a list")
                if not isinstance(item.get("autocheck_config"), dict):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.test_questions.{item_id} autocheck_config must be an object"
                    )
            coded_templates = {
                str(item.get("template_code") or "").strip().upper()
                for item in test_questions
                if isinstance(item, dict)
            }
            required_coded_templates = {"6A", "6D", "6G", "8D", "10D"}
            if len(coded_templates & required_coded_templates) < 3:
                issues.append(
                    f"intermediate_assessment.{variant_id} must include at least 3 coded template types "
                    "from 6A/6D/6G/8D/10D"
                )

            for item in open_code_questions:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.open_code_questions contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "skill_target", "student_prompt", "hidden_solution"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.open_code_questions.{item_id} missing {field}")
                if not isinstance(item.get("rubric"), list) or not item.get("rubric"):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.open_code_questions.{item_id} rubric must be a non-empty list"
                    )
                runtime_tests = item.get("runtime_tests")
                manual_rules = item.get("manual_check_rules")
                if not isinstance(runtime_tests, list):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.open_code_questions.{item_id} runtime_tests must be a list"
                    )
                if not isinstance(manual_rules, list):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.open_code_questions.{item_id} manual_check_rules must be a list"
                    )
                if not (isinstance(runtime_tests, list) and runtime_tests) and not (
                    isinstance(manual_rules, list) and manual_rules
                ):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.open_code_questions.{item_id} needs runtime_tests or manual_check_rules"
                    )
            for item in code_tasks:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "skill_target", "student_condition", "hidden_solution"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} missing {field}")
                runtime_tests = item.get("runtime_tests")
                manual_rules = item.get("manual_check_rules")
                if not isinstance(runtime_tests, list):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} runtime_tests must be a list")
                if not isinstance(manual_rules, list):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} manual_check_rules must be a list")
                if not (isinstance(runtime_tests, list) and runtime_tests) and not (
                    isinstance(manual_rules, list) and manual_rules
                ):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.code_tasks.{item_id} needs runtime_tests or manual_check_rules"
                    )
            if len(item_ids) != len(set(item_ids)):
                issues.append(f"intermediate_assessment.{variant_id} item ids must be unique")

        if len(variant_ids) != len(set(variant_ids)):
            issues.append("intermediate_assessment variant ids must be unique")
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _intermediate_matching_pairs(self, question: dict[str, Any]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for answer in question.get("correct_answers") or []:
            pair = self._split_intermediate_pair(str(answer or ""))
            if pair is not None:
                pairs.append(pair)

        autocheck = question.get("autocheck_config")
        if isinstance(autocheck, dict):
            left_by_id = self._intermediate_side_text_by_id(autocheck.get("left"))
            right_by_id = self._intermediate_side_text_by_id(autocheck.get("right"))
            for raw_pair in autocheck.get("correct_pairs") or []:
                if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                    continue
                left = left_by_id.get(str(raw_pair[0]))
                right = right_by_id.get(str(raw_pair[1]))
                if left and right:
                    pairs.append((left, right))

        return list(dict.fromkeys(pairs))

    @staticmethod
    def _split_intermediate_pair(value: str) -> tuple[str, str] | None:
        for separator in ("->", "=>", "—", "–"):
            if separator not in value:
                continue
            left, right = value.split(separator, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
        return None

    @staticmethod
    def _intermediate_side_text_by_id(items: Any) -> dict[str, str]:
        if not isinstance(items, list):
            return {}
        result: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if item_id and text:
                result[item_id] = text
        return result

    def _validate_practice_templates(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        templates: dict[str, Any],
    ) -> ValidationResult:
        expected = source_contract_for_spec(task, spec).get("tasks") or []
        actual = templates.get("tasks") if isinstance(templates, dict) else None
        issues = self._practice_task_order_issues(expected, actual, label="practice_templates")
        if isinstance(actual, list):
            for item in actual:
                if not isinstance(item, dict):
                    issues.append("practice_templates contains a non-object task")
                    continue
                for field in ("id", "level", "source_text", "task_type", "skill_target", "test_policy"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"practice_templates.{item.get('id') or '?'} missing required field {field}")
                for field in ("invariants", "slots_to_fill", "constraints"):
                    if not isinstance(item.get(field), list):
                        issues.append(f"practice_templates.{item.get('id') or '?'} field {field} must be a list")
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _validate_practice_instances(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        templates: dict[str, Any],
        instances: dict[str, Any],
    ) -> ValidationResult:
        expected = source_contract_for_spec(task, spec).get("tasks") or []
        actual = instances.get("tasks") if isinstance(instances, dict) else None
        issues = self._practice_task_order_issues(expected, actual, label="practice_instances")
        template_by_id = {
            str(item.get("id")): item
            for item in templates.get("tasks", [])
            if isinstance(item, dict) and item.get("id")
        }
        if isinstance(actual, list):
            for item in actual:
                if not isinstance(item, dict):
                    issues.append("practice_instances contains a non-object task")
                    continue
                task_id = str(item.get("id") or "?")
                for field in (
                    "id",
                    "template_id",
                    "level",
                    "task_type",
                    "scenario",
                    "student_condition",
                    "hidden_solution",
                    "teacher_explanation",
                ):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"practice_instances.{task_id} missing required field {field}")
                if item.get("template_id") != item.get("id"):
                    issues.append(f"practice_instances.{task_id} template_id must match id")
                template = template_by_id.get(task_id)
                if template is not None:
                    if item.get("level") != template.get("level"):
                        issues.append(f"practice_instances.{task_id} level does not match template")
                    if item.get("task_type") != template.get("task_type"):
                        issues.append(f"practice_instances.{task_id} task_type does not match template")
                if not isinstance(item.get("tests"), list):
                    issues.append(f"practice_instances.{task_id} tests must be a list")
                if not isinstance(item.get("runtime_tests"), list):
                    issues.append(f"practice_instances.{task_id} runtime_tests must be a list")
                if not isinstance(item.get("manual_checks"), list):
                    issues.append(f"practice_instances.{task_id} manual_checks must be a list")
                if not isinstance(item.get("subtasks"), list):
                    issues.append(f"practice_instances.{task_id} subtasks must be a list")
                if not isinstance(item.get("uniqueness_notes"), list):
                    issues.append(f"practice_instances.{task_id} uniqueness_notes must be a list")
                for field in ("faulty_code", "faulty_code_display", "display_note"):
                    if field in item and not isinstance(item.get(field), str):
                        issues.append(f"practice_instances.{task_id} {field} must be a string")
                if str(item.get("faulty_code") or "").strip() and not (
                    str(item.get("faulty_code_display") or "").strip()
                    or str(item.get("starter_code") or "").strip()
                ):
                    issues.append(
                        f"practice_instances.{task_id} has faulty_code but no learner-facing faulty_code_display or starter_code"
                    )
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _practice_task_order_issues(
        self,
        expected: Any,
        actual: Any,
        *,
        label: str,
    ) -> list[str]:
        issues: list[str] = []
        if not isinstance(expected, list):
            expected = []
        if not isinstance(actual, list):
            return [f"{label}.tasks must be a list"]
        expected_ids = [str(item.get("id")) for item in expected if isinstance(item, dict)]
        actual_ids = [str(item.get("id")) for item in actual if isinstance(item, dict)]
        if actual_ids != expected_ids:
            issues.append(f"{label} task ids/order mismatch: expected {expected_ids}, got {actual_ids}")
        expected_levels = {
            str(item.get("id")): item.get("level")
            for item in expected
            if isinstance(item, dict) and item.get("id")
        }
        for item in actual:
            if isinstance(item, dict) and item.get("id") in expected_levels and item.get("level") != expected_levels[item["id"]]:
                issues.append(f"{label}.{item['id']} level mismatch: expected {expected_levels[item['id']]}, got {item.get('level')}")
        return issues

    def _validate_with_llm(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        content: str,
        rule_result: ValidationResult,
        generation_artifacts: dict[str, Any] | None = None,
    ) -> ValidationResult:
        if not self.config.use_llm_validator:
            self.trace.log("worker.llm_validation.skipped", kind=spec.kind)
            return ValidationResult.ok()
        self.trace.log("worker.llm_validation.start", kind=spec.kind)
        validation_prompt = build_validation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            content=content,
            rule_result=rule_result,
            generation_artifacts=generation_artifacts,
        )
        data_model = self.invoker.invoke(
            "MaterialValidatorAgent",
            system=build_validator_system_prompt(),
            prompt=validation_prompt,
            schema=MaterialValidationDecision,
        )
        data = _model_to_dict(data_model)
        result = ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=_string_list(data.get("issues", [])),
            fix_instructions=_string_list(data.get("fix_instructions", data.get("issues", []))),
            issues_by_block=_list_of_dicts(data.get("issues_by_block", [])),
            passed_blocks=_list_of_dicts(data.get("passed_blocks", [])),
        )
        if spec.kind == "practice":
            filtered_result = _filter_practice_internal_reference_field_issues(result)
            if filtered_result != result:
                self.trace.log(
                    "worker.llm_validation.filtered_internal_reference_fields",
                    kind=spec.kind,
                    before_issues=result.issues,
                    after_issues=filtered_result.issues,
                )
                result = filtered_result
        self.trace.log("worker.llm_validation.done", kind=spec.kind, approved=result.approved, issues=result.issues)
        return result

    def _review_validation_failure(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        content: str,
        rule_result: ValidationResult | None,
        llm_result: ValidationResult | None,
        validation: ValidationResult | None,
        generation_artifacts: dict[str, Any] | None,
        attempt_store: AttemptArtifactStore,
    ) -> dict[str, Any]:
        if not self.config.use_llm_validator or not self.config.use_validation_controller:
            return {}
        if not content or rule_result is None or llm_result is None or validation is None:
            return {}
        if not rule_result.approved:
            self.trace.log("worker.controller.skipped_rule_failed", kind=spec.kind, issues=rule_result.issues)
            return {}

        self.trace.log("worker.controller.start", kind=spec.kind, issues=validation.issues)
        prompt = build_validation_controller_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            content=content,
            rule_result=rule_result,
            llm_result=llm_result,
            merged_validation=validation,
            generation_artifacts=generation_artifacts,
        )
        data_model = self.invoker.invoke(
            "ValidationControllerAgent",
            system=build_validation_controller_system_prompt(),
            prompt=prompt,
            schema=ValidationControllerDecision,
        )
        data = _model_to_dict(data_model)
        decision = {
            "approved": bool(data.get("approved")),
            "decision": str(data.get("decision") or ("approve_material" if data.get("approved") else "keep_failed")),
            "quality_score": _controller_quality_score(data),
            "score_rationale": str(data.get("score_rationale") or ""),
            "rationale": str(data.get("rationale") or ""),
            "blocking_issues": _string_list(data.get("blocking_issues", [])),
            "non_blocking_issues": _string_list(data.get("non_blocking_issues", [])),
            "overruled_validator_issues": _string_list(data.get("overruled_validator_issues", [])),
            "residual_risks": _string_list(data.get("residual_risks", [])),
            "fix_instructions": _string_list(data.get("fix_instructions", [])),
        }
        decision = self._apply_intermediate_appellate_policy(
            spec=spec,
            rule_result=rule_result,
            validation=validation,
            generation_artifacts=generation_artifacts,
            decision=decision,
        )
        decision = self._apply_practice_appellate_policy(
            spec=spec,
            task=task,
            rule_result=rule_result,
            validation=validation,
            generation_artifacts=generation_artifacts,
            decision=decision,
        )
        decision = self._apply_specification_qa_appellate_policy(
            spec=spec,
            rule_result=rule_result,
            validation=validation,
            decision=decision,
        )
        decision = self._apply_mr_intermediate_appellate_policy(
            spec=spec,
            content=content,
            rule_result=rule_result,
            validation=validation,
            decision=decision,
        )
        self.trace.log(
            "worker.controller.done",
            kind=spec.kind,
            approved=decision["approved"],
            quality_score=decision["quality_score"],
            accept_score=self.config.validation_controller_accept_score,
            accepted_by_score=decision["quality_score"] >= self.config.validation_controller_accept_score,
            blocking_issues=decision["blocking_issues"],
            non_blocking_issues=decision["non_blocking_issues"],
        )
        attempt_store.write_material_controller_review(
            kind=spec.kind,
            content=content,
            controller_decision=decision,
            metadata={
                "agent_type": spec.agent_type,
                "material_type": spec.material_type,
                "validator_issues": validation.issues,
            },
        )
        return decision

    def _apply_intermediate_appellate_policy(
        self,
        *,
        spec: MaterialSpec,
        rule_result: ValidationResult,
        validation: ValidationResult,
        generation_artifacts: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if spec.kind != "intermediate" or not rule_result.approved:
            return decision
        if not self._intermediate_artifact_approved(generation_artifacts):
            return decision

        blocking_issues = [str(item) for item in (decision.get("blocking_issues") or validation.issues or [])]
        if not blocking_issues:
            return decision

        overruled: list[str] = []
        remaining: list[str] = []
        for issue in blocking_issues:
            if self._is_overstrict_intermediate_issue(issue):
                overruled.append(issue)
            else:
                remaining.append(issue)

        if remaining or not overruled:
            return decision

        adjusted = dict(decision)
        adjusted["approved"] = True
        adjusted["decision"] = "approve_material"
        adjusted["quality_score"] = max(
            float(adjusted.get("quality_score", 0.0) or 0.0),
            self.config.validation_controller_accept_score,
        )
        note = (
            "Deterministic appellate policy overruled intermediate validator objections that confused "
            "candidate answer options, publishable HTML, or error-fixing prompts with visible answer-key leaks."
        )
        adjusted["score_rationale"] = " ".join(part for part in [str(adjusted.get("score_rationale") or ""), note] if part)
        adjusted["rationale"] = " ".join(part for part in [str(adjusted.get("rationale") or ""), note] if part)
        adjusted["blocking_issues"] = []
        adjusted["non_blocking_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("non_blocking_issues", [])), *overruled])
        )
        adjusted["overruled_validator_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("overruled_validator_issues", [])), *overruled])
        )
        adjusted["fix_instructions"] = [
            item
            for item in _string_list(adjusted.get("fix_instructions", []))
            if item not in set(overruled)
        ]
        return adjusted

    def _apply_practice_appellate_policy(
        self,
        *,
        spec: MaterialSpec,
        task: dict[str, Any],
        rule_result: ValidationResult,
        validation: ValidationResult,
        generation_artifacts: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if spec.kind != "practice" or not rule_result.approved:
            return decision

        contract = source_contract_for_spec(task, spec)
        authoritative_ids = [str(item) for item in contract.get("authoritative_task_ids", []) if str(item)]
        if not authoritative_ids:
            return decision

        artifact_ids = self._practice_artifact_task_ids(generation_artifacts)
        if artifact_ids and artifact_ids != authoritative_ids:
            return decision
        if not artifact_ids:
            return decision

        blocking_issues = [str(item) for item in (decision.get("blocking_issues") or validation.issues or [])]
        if not blocking_issues:
            return decision

        overruled: list[str] = []
        remaining: list[str] = []
        for issue in blocking_issues:
            if self._is_overstrict_practice_task_count_issue(issue, authoritative_ids) or (
                self._is_overstrict_practice_faulty_code_issue(issue)
            ) or (
                self._is_overstrict_practice_subject_entity_issue(issue)
            ) or (
                self._is_overstrict_practice_formatting_issue(issue)
            ):
                overruled.append(issue)
            else:
                remaining.append(issue)

        if not overruled:
            return decision

        adjusted = dict(decision)
        note = (
            "Deterministic appellate policy overruled practice validator objections that used an over-narrow "
            "interpretation of the practice contract: lesson.practice_tasks/authoritative_task_ids define the "
            "task set, intentionally faulty code may be invalid by design, and source subject entities are slot "
            "examples unless exact entities are explicitly required. Practice validation checks methodology and "
            "topic coverage rather than requiring one rigid task rendering structure."
        )
        adjusted["score_rationale"] = " ".join(part for part in [str(adjusted.get("score_rationale") or ""), note] if part)
        adjusted["rationale"] = " ".join(part for part in [str(adjusted.get("rationale") or ""), note] if part)
        adjusted["blocking_issues"] = remaining
        adjusted["non_blocking_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("non_blocking_issues", [])), *overruled])
        )
        adjusted["overruled_validator_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("overruled_validator_issues", [])), *overruled])
        )
        adjusted["fix_instructions"] = [
            item
            for item in _string_list(adjusted.get("fix_instructions", []))
            if item not in set(overruled)
            and not self._is_overstrict_practice_task_count_issue(item, authoritative_ids)
            and not self._is_overstrict_practice_faulty_code_issue(item)
            and not self._is_overstrict_practice_subject_entity_issue(item)
            and not self._is_overstrict_practice_formatting_issue(item)
        ]
        if not remaining:
            adjusted["approved"] = True
            adjusted["decision"] = "approve_material"
            adjusted["quality_score"] = max(
                float(adjusted.get("quality_score", 0.0) or 0.0),
                self.config.validation_controller_accept_score,
            )
        return adjusted

    @staticmethod
    def _practice_artifact_task_ids(generation_artifacts: dict[str, Any] | None) -> list[str]:
        if not isinstance(generation_artifacts, dict):
            return []
        instances = generation_artifacts.get("practice_instances")
        if not isinstance(instances, dict):
            return []
        tasks = instances.get("tasks")
        if not isinstance(tasks, list):
            return []
        return [str(item.get("id")) for item in tasks if isinstance(item, dict) and item.get("id")]

    def _apply_specification_qa_appellate_policy(
        self,
        *,
        spec: MaterialSpec,
        rule_result: ValidationResult,
        validation: ValidationResult,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if spec.kind != "specification_qa" or not rule_result.approved:
            return decision

        blocking_issues = [str(item) for item in (decision.get("blocking_issues") or validation.issues or [])]
        if not blocking_issues:
            return decision

        overruled: list[str] = []
        remaining: list[str] = []
        for issue in blocking_issues:
            if self._is_overstrict_specification_qa_id_issue(issue):
                overruled.append(issue)
            else:
                remaining.append(issue)

        if not overruled:
            return decision

        adjusted = dict(decision)
        note = (
            "Deterministic appellate policy overruled validator objections that treated visible QA-ID labels "
            "as leakage in specification_qa. QA-ID is allowed for this internal QA artifact."
        )
        adjusted["score_rationale"] = " ".join(part for part in [str(adjusted.get("score_rationale") or ""), note] if part)
        adjusted["rationale"] = " ".join(part for part in [str(adjusted.get("rationale") or ""), note] if part)
        adjusted["blocking_issues"] = remaining
        adjusted["non_blocking_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("non_blocking_issues", [])), *overruled])
        )
        adjusted["overruled_validator_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("overruled_validator_issues", [])), *overruled])
        )
        adjusted["fix_instructions"] = [
            item
            for item in _string_list(adjusted.get("fix_instructions", []))
            if not self._mentions_qa_id(item)
        ]
        if not remaining:
            adjusted["approved"] = True
            adjusted["decision"] = "approve_material"
            adjusted["quality_score"] = max(
                float(adjusted.get("quality_score", 0.0) or 0.0),
                self.config.validation_controller_accept_score,
            )
        return adjusted

    def _apply_mr_intermediate_appellate_policy(
        self,
        *,
        spec: MaterialSpec,
        content: str,
        rule_result: ValidationResult,
        validation: ValidationResult,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if spec.kind != "mr_intermediate" or not rule_result.approved:
            return decision
        if self._mr_intermediate_visible_internal_markers(content):
            return decision

        blocking_issues = [str(item) for item in (decision.get("blocking_issues") or validation.issues or [])]
        if not blocking_issues:
            return decision

        overruled: list[str] = []
        remaining: list[str] = []
        for issue in blocking_issues:
            if self._is_overstrict_mr_intermediate_kim_issue(issue):
                overruled.append(issue)
            else:
                remaining.append(issue)

        if not overruled:
            return decision

        adjusted = dict(decision)
        note = (
            "Deterministic appellate policy overruled mr_intermediate validator objections that treated "
            "dependency intermediate content as duplicated publishable KIM content."
        )
        adjusted["score_rationale"] = " ".join(part for part in [str(adjusted.get("score_rationale") or ""), note] if part)
        adjusted["rationale"] = " ".join(part for part in [str(adjusted.get("rationale") or ""), note] if part)
        adjusted["blocking_issues"] = remaining
        adjusted["non_blocking_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("non_blocking_issues", [])), *overruled])
        )
        adjusted["overruled_validator_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("overruled_validator_issues", [])), *overruled])
        )
        adjusted["fix_instructions"] = [
            item
            for item in _string_list(adjusted.get("fix_instructions", []))
            if item not in set(overruled)
            and not self._is_overstrict_mr_intermediate_kim_issue(item)
        ]
        if not remaining:
            adjusted["approved"] = True
            adjusted["decision"] = "approve_material"
            adjusted["quality_score"] = max(
                float(adjusted.get("quality_score", 0.0) or 0.0),
                self.config.validation_controller_accept_score,
            )
        return adjusted

    @staticmethod
    def _mr_intermediate_visible_internal_markers(content: str) -> list[str]:
        normalized = _normalize_copy_text(content)
        markers = (
            "intermediate_assessment",
            "generation_artifacts",
            "hidden_solution",
            "autocheck_config",
        )
        return [marker for marker in markers if marker in normalized]

    @staticmethod
    def _is_overstrict_mr_intermediate_kim_issue(issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if not normalized:
            return False
        if any(
            marker in normalized
            for marker in (
                "intermediate_assessment",
                "generation_artifacts",
                "hidden_solution",
                "autocheck_config",
            )
        ):
            return False
        duplication_markers = (
            "duplicate",
            "duplicates",
            "duplication",
            "full",
            "bank",
            "kim",
            "assessment",
            "\u0434\u0443\u0431\u043b",
            "\u043f\u043e\u043b\u043d",
            "\u0431\u0430\u043d\u043a",
            "\u043a\u0438\u043c",
        )
        item_markers = (
            "variant",
            "variants",
            "task",
            "tasks",
            "question",
            "questions",
            "\u0432\u0430\u0440\u0438\u0430\u043d\u0442",
            "\u0437\u0430\u0434\u0430\u043d",
            "\u0432\u043e\u043f\u0440\u043e\u0441",
        )
        return any(marker in normalized for marker in duplication_markers) and any(
            marker in normalized for marker in item_markers
        )

    @staticmethod
    def _is_overstrict_practice_task_count_issue(issue: str, authoritative_ids: list[str]) -> bool:
        normalized = _normalize_copy_text(issue)
        if not normalized:
            return False
        count_markers = (
            "количеств",
            "требуется",
            "фактически",
            "отсутств",
            "missing",
            "required",
            "count",
            "задач",
            "task",
        )
        if not any(marker in normalized for marker in count_markers):
            return False
        if not any(marker in normalized for marker in ("difficulty", "l1", "l2", "уров", "p6", "p7")):
            return False
        authoritative = {_normalize_copy_text(task_id) for task_id in authoritative_ids}
        mentioned_p_ids = set(re.findall(r"\bp\d+\b", normalized))
        non_authoritative_p_ids = mentioned_p_ids - authoritative
        if non_authoritative_p_ids:
            return True
        return any(marker in normalized for marker in ("difficulty", "l1", "l2", "уров")) and "practice_tasks" not in normalized

    @staticmethod
    def _is_overstrict_practice_faulty_code_issue(issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if not normalized:
            return False
        faulty_context = (
            "faulty_code",
            "faulty_code_display",
            "ошибочный код",
            "faulty",
            "фрагмент",
            "код",
        )
        unclosed_string_context = (
            "незакрыт",
            "незаверш",
            "unterminated",
            "unclosed",
            "eol while scanning",
            "string literal",
            "строк",
            "кавыч",
        )
        overstrict_markers = (
            "разрыва",
            "две строки",
            "следующ",
            "многостроч",
            "multi-line",
            "next line",
            "spans",
            "invalid",
            "невалид",
            "одна ошибка",
            "one error",
            "структур",
            "скобк",
            "parse",
            "парс",
            "восстановлен",
        )
        exact_fix_or_answer_markers = (
            "исправленный код",
            "corrected code",
            "hidden_solution",
            "exact fix",
            "точная правка",
            "reveals the fix",
            "раскрывает",
        )
        if any(marker in normalized for marker in exact_fix_or_answer_markers):
            return False
        return (
            any(marker in normalized for marker in faulty_context)
            and any(marker in normalized for marker in unclosed_string_context)
            and any(marker in normalized for marker in overstrict_markers)
        )

    @staticmethod
    def _is_overstrict_practice_subject_entity_issue(issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if not normalized:
            return False
        subject_entity_markers = (
            "сущност",
            "предметн",
            "слот",
            "slot",
            "entity",
            "entities",
            "категор",
            "любимый цвет",
            "любимое животное",
            "favorite color",
            "favorite animal",
            "напиток",
            "спорт",
            "drink",
            "sport",
        )
        overstrict_markers = (
            "подмен",
            "замен",
            "измен",
            "вместо",
            "replace",
            "replacing",
            "instead",
            "source_text",
            "паттерн",
            "pattern",
        )
        hard_failure_markers = (
            "другой навык",
            "другой тип",
            "different skill",
            "different task type",
            "wrong task type",
            "не тот тип",
            "не тот навык",
        )
        exact_required_markers = (
            "явно требует",
            "explicitly requires",
            "exact entities",
            "дословн",
            "точно эти",
        )
        if any(marker in normalized for marker in hard_failure_markers):
            return False
        if any(marker in normalized for marker in exact_required_markers):
            return False
        return any(marker in normalized for marker in subject_entity_markers) and any(
            marker in normalized for marker in overstrict_markers
        )

    @staticmethod
    def _is_overstrict_practice_formatting_issue(issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if not normalized:
            return False
        formatting_markers = (
            "<pre><code>",
            "pre code",
            "code block",
            "кодовый блок",
            "блок код",
            "раздел код",
            "подблок код",
            "код в редакторе",
            "starter code",
            "starter_code",
            "заготовк",
            "placeholder",
            "место для кода",
            "оформлен",
            "структур",
            "layout",
            "formatting",
        )
        methodology_failure_markers = (
            "невыполним",
            "не может выполнить",
            "нет проверки",
            "не проверяется",
            "no checking",
            "uncheckable",
            "incoherent",
            "противореч",
            "contradict",
            "раскрывает ответ",
            "answer leakage",
            "hidden_solution",
            "corrected code",
        )
        if any(marker in normalized for marker in methodology_failure_markers):
            return False
        return any(marker in normalized for marker in formatting_markers)

    def _intermediate_artifact_approved(self, generation_artifacts: dict[str, Any] | None) -> bool:
        if not isinstance(generation_artifacts, dict):
            return False
        check = generation_artifacts.get("intermediate_assessment_check")
        return isinstance(check, dict) and bool(check.get("approved"))

    def _is_overstrict_intermediate_issue(self, issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if any(
            marker in normalized
            for marker in (
                "correct_answers",
                "reference_answer",
                "autocheck_config",
                "hidden_solution",
                "teacher_explanation",
                "internal_explanation",
            )
        ):
            return False

        coded_template_tokens = ("10d", "6a", "6d", "6g", "8d")
        if "10d" in normalized and any(token in normalized for token in ("утеч", "ключ", "ответ", "answer", "key")):
            return True
        if any(token in normalized for token in coded_template_tokens) and any(
            token in normalized for token in ("шаблон", "template", "размет", "markup", "html")
        ):
            return True
        if any(token in normalized for token in ("исправ", "fix")) and any(
            token in normalized for token in ("решен", "solution", "эталон", "ключ", "key")
        ):
            return True
        return False

    def _is_overstrict_specification_qa_id_issue(self, issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if not self._mentions_qa_id(issue):
            return False
        leakage_markers = (
            "leak",
            "leakage",
            "internal marker",
            "service marker",
            "source marker",
            "утеч",
            "служеб",
            "маркер",
            "внутрен",
            "идентификатор",
        )
        return any(marker in normalized for marker in leakage_markers)

    def _mentions_qa_id(self, text: str) -> bool:
        normalized = _normalize_copy_text(text)
        return any(marker in normalized for marker in ("qa-id", "qa id", "qa_id"))


class PackageValidator:
    def __init__(
        self,
        *,
        subagents: Mapping[str, Any],
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
        trace: TraceLogger | None = None,
    ) -> None:
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()
        self.trace = trace or TraceLogger()
        self.invoker = StructuredSubagentInvoker(
            subagents,
            trace=self.trace,
            langchain_config=config.langchain_config,
        )

    def validate(
        self,
        *,
        task: dict[str, Any],
        specs: list[MaterialSpec],
        materials: list[MaterialResult],
        attempts_dir: Path | None = None,
    ) -> ValidationResult:
        attempt_store = AttemptArtifactStore(attempts_dir)
        self.trace.log(
            "package_validation.start",
            material_count=len(materials),
            material_statuses=[{"kind": item.kind, "status": item.status} for item in materials],
        )
        rule_result = self.rule_validator.validate_package(specs, materials)
        self.trace.log("package_validation.rule.done", approved=rule_result.approved, issues=rule_result.issues)
        llm_result: ValidationResult | None = None
        if not self.config.use_llm_validator:
            self.trace.log("package_validation.llm.skipped")
            advisory_result = ValidationResult(
                approved=True,
                issues=list(rule_result.issues),
                fix_instructions=list(rule_result.fix_instructions),
                issues_by_block=list(rule_result.issues_by_block),
                passed_blocks=list(rule_result.passed_blocks),
            )
            attempt_store.write_package_validation(
                rule_result=rule_result,
                llm_result=None,
                validation=advisory_result,
                metadata={"material_count": len(materials), "advisory": True},
            )
            return advisory_result

        prompt = build_package_validation_prompt(
            task=task,
            specs=specs,
            materials=materials,
            rule_result=rule_result,
        )
        package_system_prompt = (
            "You are PackageValidatorAgent. Check the package only. "
            "Do not generate or repair materials. Return structured validation fields."
        )
        data_model = self.invoker.invoke(
            "PackageValidatorAgent",
            system=package_system_prompt,
            prompt=prompt,
            schema=PackageValidationDecision,
        )
        data = _model_to_dict(data_model)
        llm_result = ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=_string_list(data.get("issues", [])),
            fix_instructions=_string_list(data.get("fix_instructions", data.get("issues", []))),
        )
        merged = rule_result.merge(llm_result)
        result = ValidationResult(
            approved=True,
            issues=list(merged.issues),
            fix_instructions=list(merged.fix_instructions),
            issues_by_block=list(merged.issues_by_block),
            passed_blocks=list(merged.passed_blocks),
        )
        self.trace.log("package_validation.done", approved=result.approved, issues=result.issues, advisory=True)
        attempt_store.write_package_validation(
            rule_result=rule_result,
            llm_result=llm_result,
            validation=result,
            metadata={"material_count": len(materials), "advisory": True},
        )
        return result
