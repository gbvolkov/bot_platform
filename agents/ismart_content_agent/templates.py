from __future__ import annotations

from .contracts import InteractiveCode, TemplateId, TemplateSpec


TEMPLATE_REGISTRY: dict[TemplateId, TemplateSpec] = {
    "theory": TemplateSpec(
        template_id="theory",
        content_type="theory",
        required_fields=["sections", "summary"],
    ),
    "practice_python": TemplateSpec(
        template_id="practice_python",
        content_type="practice",
        required_fields=[
            "module_number",
            "topic",
            "assignment_title",
            "goal",
            "condition",
            "input_data",
            "expected_result",
            "criteria",
            "difficulty_level",
            "assessment_alignment",
            "source_binding",
        ],
        service_fields=["service_solution", "tests"],
    ),
    "self_study": TemplateSpec(
        template_id="self_study",
        content_type="self_study",
        required_fields=["task", "algorithm", "submission_requirements"],
        service_fields=["answer_key"],
    ),
    "control_question": TemplateSpec(
        template_id="control_question",
        content_type="control_question",
        required_fields=["question", "answer_format"],
        service_fields=["answer_key"],
    ),
    "interactive_template": TemplateSpec(
        template_id="interactive_template",
        content_type="interactive_template",
        required_fields=["template_code", "prompt"],
        service_fields=["answer_key"],
        preview_required=True,
        supported_interactive_codes=["6A", "6D", "6G", "8D", "10D", "3H", "3D"],
    ),
}


def get_template(template_id: TemplateId) -> TemplateSpec:
    try:
        return TEMPLATE_REGISTRY[template_id]
    except KeyError as exc:
        raise ValueError(f"Unknown template_id: {template_id}") from exc


def validate_interactive_code(code: InteractiveCode | None) -> InteractiveCode:
    if code is None:
        raise ValueError("interactive_template requires interactive_code")
    supported = TEMPLATE_REGISTRY["interactive_template"].supported_interactive_codes
    if code not in supported:
        raise ValueError(f"Unsupported interactive_code: {code}")
    return code
