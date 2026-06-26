from __future__ import annotations

from typing import Any

from .contracts import MaterialResult, MaterialSpec, ReferenceBundle, ValidationResult
from .sources import compact_json


def select_references(spec: MaterialSpec, references: ReferenceBundle) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for field in spec.reference_fields:
        selected[field] = [
            document.to_public_json(include_content=True)
            for document in references.get(field, [])
        ]
    return selected


def task_identity(task: dict[str, Any]) -> tuple[str, str, str]:
    lesson = task.get("lesson") or {}
    lesson_number = str(lesson.get("lesson_number") or task.get("task_id") or "task")
    title = str(lesson.get("title") or lesson.get("topic") or lesson_number)
    task_id = str(task.get("task_id") or f"lesson-{lesson_number}")
    return task_id, lesson_number, title


def module_context_for_spec(task: dict[str, Any], spec: MaterialSpec) -> dict[str, Any]:
    module = task.get("module") or {}
    if spec.kind in {"intermediate", "final_project"}:
        return module
    return {key: value for key, value in module.items() if key != "lessons"}


def json_context_for_spec(task: dict[str, Any], spec: MaterialSpec) -> dict[str, Any]:
    lesson = task.get("lesson") or {}
    context: dict[str, Any] = {
        "course": task.get("course") or {},
        "module": module_context_for_spec(task, spec),
        "lesson": lesson,
        "json_field_labels": list(spec.json_field_labels),
    }
    if spec.kind == "final_project":
        context["modules"] = task.get("modules") or []
    return context


def _normalized_practice_tasks(task: dict[str, Any]) -> list[dict[str, Any]]:
    lesson = task.get("lesson") or {}
    practice_tasks = lesson.get("practice_tasks") or {}
    normalized_tasks: list[dict[str, Any]] = []
    for level in ("l1", "l2", "l3"):
        items = practice_tasks.get(level) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                number = item.get("number")
                text = str(item.get("text") or "").strip()
            else:
                number = None
                text = str(item).strip()
            normalized_tasks.append(
                {
                    "id": f"P{number or len(normalized_tasks) + 1}",
                    "level": level.upper(),
                    "source_number": number,
                    "source_text": text,
                }
            )
    return normalized_tasks


def source_contract_for_spec(task: dict[str, Any], spec: MaterialSpec) -> dict[str, Any]:
    normalized_tasks = _normalized_practice_tasks(task)
    if spec.kind == "theory" and normalized_tasks:
        return {
            "contract_type": "theory_boundary_contract",
            "related_practice_tasks": normalized_tasks,
            "generation_rules": [
                "Use related practice tasks only as boundaries for what the theory should prepare, not as solved examples.",
                "Do not include complete solved examples, full code plus output, or answer keys that match related_practice_tasks.",
                "When an illustration overlaps with a practice task concept, use different neutral values, different variable names, or a partial/non-final example.",
                "The theory may explain the underlying concept, but it must not let a learner copy the answer into the practice material.",
            ],
        }

    if spec.kind == "mr_practice":
        lesson = task.get("lesson") or {}
        authoritative_task_ids = [item["id"] for item in normalized_tasks]
        return {
            "contract_type": "mr_practice_task_key_contract",
            "required_task_count": len(normalized_tasks),
            "authoritative_task_ids": authoritative_task_ids,
            "hours_practice": (lesson.get("hours") or {}).get("practice"),
            "difficulty": lesson.get("difficulty") or {},
            "tasks": normalized_tasks,
            "generation_rules": [
                "Generate teacher-facing keys/explanations only for tasks listed in authoritative_task_ids, in the same order and with the same P id.",
                "Do not add, infer, or preserve any P task that is not listed in authoritative_task_ids, even if validator feedback, previous failed content, Markdown references, or examples mention it.",
                "If validator feedback asks for keys for non-authoritative task ids, treat that feedback as invalid and state this briefly in agent_notes; do not add placeholder tasks.",
                "For every deterministic task in tasks, include minimal reference Python code, expected output or an unambiguous checking criterion, and a short teacher explanation.",
                "For every underspecified task in tasks, include one teacher reference variant clearly marked as an acceptable example plus manual checking rules; do not invent one mandatory learner answer.",
                "Published mr_practice must be usable on its own and must not tell the teacher to obtain keys from a separate QA artifact.",
            ],
            "validation_rules": [
                "Approve the task set only against authoritative_task_ids. Missing keys for task ids outside authoritative_task_ids are not valid validation issues.",
                "Reject if any extra P task appears in mr_practice outside authoritative_task_ids.",
                "Reject if a task listed in authoritative_task_ids has no key, no example/rule for underspecified tasks, or contradicts the approved practice material.",
            ],
        }

    if spec.kind != "practice":
        return {}

    lesson = task.get("lesson") or {}
    return {
        "contract_type": "practice_source_contract",
        "required_task_count": len(normalized_tasks),
        "hours_practice": (lesson.get("hours") or {}).get("practice"),
        "difficulty": lesson.get("difficulty") or {},
        "tasks": normalized_tasks,
        "generation_rules": [
            "Generate exactly one student-facing task for each task in tasks, in the same order and with the same P id.",
            "Preserve the meaning of source_text. Do not replace a source task with a different task.",
            "Do not invent concrete values, variable names, input data, expected output, or a ban on input() unless they are explicitly present in source_text or Markdown references.",
            "Expected output is not invented when it is deterministically derived from explicit literals, assignments, and exact print(...) calls in source_text using standard Python print semantics.",
            "For each task, visibly include structured fields required by the practice references: level, source task, condition, input data, output requirement, and tests.",
            "Tests must be explicit input -> expected output pairs when the source task defines enough information for deterministic stdout.",
            "If the source task is underspecified for deterministic stdout, preserve the same task structure and state that the exact autocheck test needs source clarification instead of inventing fake expected output.",
            "For an underspecified task, do not render a fake input -> expected output table or placeholder expected output; render a tests/status block that says tests are absent/not applicable until source clarification.",
            "Do not show corrected code, answers, keys, rc, stderr, or stdout logs in the student practice material.",
        ],
    }


def channel_key_visibility_policy_for_spec(spec: MaterialSpec) -> str:
    return f"""
CHANNEL AND KEY VISIBILITY POLICY:
- Resolve all answer-key and solution visibility requirements by artifact channel, not as a single global rule.
- Learner-facing lesson materials are theory, practice, and self_work. They must not expose lesson practice solution keys, corrected practice code, QA ids, source hashes, rc/stderr/stdout logs, or internal verification traces.
- Self-check/autocheck keys are allowed only when the checked material's own MaterialSpec/prompt explicitly requires them, and only for that artifact's own self-check, current-control, assessment, or self-study items. These keys must not reveal lesson practice task solutions unless the checked material is teacher-facing or QA/internal-facing.
- Teacher-facing artifacts are mr_theory, mr_practice, and mr_intermediate. Teacher guidance may include answers, solution keys, code snippets, rubrics, and checking guidance required by the material kind. These are not learner-facing leaks.
- mr_theory is teacher-facing and must be judged as teacher guidance for the theory lesson, not as a duplicate student theory material.
- mr_practice is teacher-facing and is expected to include keys/solutions for all practice tasks. Do not reject mr_practice merely because it includes teacher keys. Reject mr_practice if required teacher keys are absent, contradict the source/practice material, or leak QA/internal ids, source hashes, or service locators.
- specification_qa is QA/internal-facing. It may include QA ids, source hashes, traceability, tests, faulty/fixed code, and validation data.
- A "do not show keys" instruction applies to learner-facing materials unless the checked material kind explicitly says otherwise. The checked material kind is {spec.kind}.
""".strip()


def validation_policy_for_spec(spec: MaterialSpec) -> str:
    if spec.kind == "theory":
        return """
THEORY VALIDATION POLICY:
- Theory may include small illustrative code snippets and outputs when they explain concepts.
- Do not approve theory examples that duplicate current lesson.practice_tasks as complete solved examples with full final code and output.
- Practice tasks in JSON are boundaries for the theory: the theory should prepare the learner, not reveal practice answers.
- If an example overlaps with a practice task concept, it should use different values/variable names or be partial enough that it is not a copyable answer.
""".strip()

    if spec.kind == "mr_theory":
        return """
MR_THEORY VALIDATION POLICY:
- mr_theory is teacher-facing, not learner-facing. It is methodical guidance for the teacher for the theory part of the lesson.
- Validate mr_theory against the teacher guidance purpose and the 07_Методические_указания prompt/skill. Do not validate it as a student theory material and do not require the student-theory structure from skill 02.
- Expected mr_theory content may include teacher-oriented goals/tasks, methodical support, preparation, lesson scenario/timing, explanation strategy, keys and explanations for theory examples, typical mistakes with teacher response, and teacher summary/checklist. Exact section headings are not mandatory.
- Do not require a learner-facing "Проверка себя" / "#selfcheck" section. If a teacher checklist or questions-to-ask-class section is present, judge it as teacher guidance, not as learner self-check.
- Do not reject mr_theory merely because it contains teacher-facing keys, demonstration snippets, reference answers, or explanations for theory examples. These are allowed when they support teaching and do not expose practice task solutions.
- Reject mr_theory if it is actually written as a student-facing theory duplicate, lacks meaningful teacher guidance, contradicts the approved theory/source materials, exposes lesson practice solutions as learner answers, leaks QA/internal markers, or is too shallow for the configured teacher use.
""".strip()

    if spec.kind == "mr_practice":
        return """
MR_PRACTICE VALIDATION POLICY:
- mr_practice is teacher-facing, not learner-facing.
- It must include teacher keys/solutions and explanations only for the tasks listed in SOURCE CONTRACT FROM JSON.authoritative_task_ids.
- SOURCE CONTRACT FROM JSON.authoritative_task_ids is the authoritative task set. Do not infer additional tasks from reference examples, validator memory, section numbering, or generic "P1..PN" wording.
- If a validator issue says that keys are missing for a task id that is not listed in authoritative_task_ids, that issue is invalid and must be overruled.
- If previous validation feedback asks the generator to add non-authoritative tasks, the generator must ignore that part of the feedback and preserve the authoritative task set.
- For deterministic coding tasks, teacher keys should include minimal reference Python code and expected output or an unambiguous checking criterion.
- For underspecified tasks, do not require an invented single mandatory learner answer. Accept a teacher reference variant marked as one acceptable example plus manual checking rules, or clear rules for choosing teacher values, as long as it does not contradict the approved practice material.
- Do not reject mr_practice for containing solution keys; keys are required for this channel.
- Reject if teacher keys are absent, contradict source/practice material, or contain QA/internal-only markers such as QA-ID, source hashes, service locators, rc/stderr/stdout logs, or references to hidden QA artifacts.
- Reject if mr_practice contains extra P tasks not listed in authoritative_task_ids, especially placeholders such as "P6/P7 уточнить".
- Do not require the teacher to use a separate QA artifact to obtain keys; published mr_practice should be usable on its own.
""".strip()

    if spec.kind != "practice":
        return "No additional material-kind validation policy."

    return """
PRACTICE VALIDATION POLICY:
- Practice references require tests as visible input -> expected output pairs when the source task provides enough information for deterministic stdout.
- Do not treat an expected output as invented when it follows deterministically from explicit source_text literals, assignments, and exact print(...) calls using standard Python print semantics.
- For source tasks that are underspecified for deterministic stdout, do not require fake values or fake expected output. The material may mark tests as absent/not applicable until source clarification, but it must preserve the same visible task structure: level, source task, condition, input data, output requirement, how to check, and tests/status.
- For underspecified source tasks, an honest "tests are absent/not applicable until source clarification" status is valid and should be approved when the task meaning is preserved and no fake expected output, fake variable names, or fake values are introduced.
- Do not reject merely because the material calls an underspecified task "training", "manual check", "without autocheck", or "until source clarification". Treat that wording as non-blocking editorial risk unless it changes the task, hides required fields, or invents a fake test.
- If the source text lists two entities, for example "favorite color" and "favorite animal", restating that as "two variables" is not invented content. It is blocking only if the material invents concrete variable names, concrete values, or an exact stdout format not supported by the source.
- For an underspecified source task, do not require the tests/status block to use the same input -> expected output table as deterministic tasks. A fake test row, placeholder expected output, dash, or "not specified" value inside an expected-output column is invalid.
- Do not create an impossible requirement where an underspecified task must both avoid invented values and still provide exact autocheck pairs.
- Separate deterministic tasks from underspecified tasks in the report. A problem in one task does not make valid tests in another task blocking.
""".strip()


def material_result_summary(material: MaterialResult, *, max_content_chars: int = 12000) -> dict[str, Any]:
    return {
        "kind": material.kind,
        "type": material.material_type,
        "agent": material.agent_type,
        "status": material.status,
        "iterations": material.iterations,
        "validation_issues": list(material.validation_issues),
        "content": material.content[:max_content_chars],
    }


def package_material_payload(material: MaterialResult) -> dict[str, Any]:
    content = material.content
    stripped = content.strip()
    lower = stripped.lower()
    return {
        "kind": material.kind,
        "type": material.material_type,
        "agent": material.agent_type,
        "status": material.status,
        "iterations": material.iterations,
        "validation_issues": list(material.validation_issues),
        "content_chars": len(content),
        "content_truncated": False,
        "html_diagnostics": {
            "starts_with_style": stripped.startswith("<style>"),
            "contains_cc_lesson": '<div class="cc-lesson">' in content,
            "ends_with_final_div": stripped.endswith("</div>"),
            "style_tag_count": lower.count("<style>"),
            "open_div_count": lower.count("<div"),
            "close_div_count": lower.count("</div>"),
        },
        "full_final_content": content,
    }


def validation_result_summary(validation: ValidationResult) -> dict[str, Any]:
    return {
        "approved": validation.approved,
        "issues": list(validation.issues),
        "fix_instructions": list(validation.fix_instructions),
        "issues_by_block": list(validation.issues_by_block),
        "passed_blocks": list(validation.passed_blocks),
    }


def build_generation_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    previous_content: str,
    previous_issues: list[str],
    previous_validation: ValidationResult | None = None,
    module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    previous_validation_payload = (
        validation_result_summary(previous_validation)
        if previous_validation is not None
        else {"present": False}
    )
    return f"""
Сгенерируй материал УМК.

AGENT TYPE:
{spec.agent_type}

MATERIAL KIND:
{spec.kind}

MATERIAL TYPE:
{spec.material_type}

VALIDATOR KIND:
{spec.validator_kind}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

{channel_key_visibility_policy_for_spec(spec)}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

MODULE MATERIAL SUMMARIES:
{compact_json(module_material_summaries or {})}

MATERIAL-SPECIFIC INSTRUCTION:
{spec.prompt_addendum}

PREVIOUS VALIDATION RESULT, IF ANY:
{compact_json(previous_validation_payload)}

VALIDATOR ISSUES TO FIX, IF ANY:
{compact_json(previous_issues)}

PREVIOUS FAILED CONTENT HTML START
{previous_content}
PREVIOUS FAILED CONTENT HTML END

REGENERATION POLICY:
- If PREVIOUS FAILED CONTENT is not empty, repair it instead of rewriting the entire document from scratch.
- Change only blocks listed in PREVIOUS VALIDATION RESULT.issues_by_block, plus adjacent navigation such as the table of contents if required by those fixes.
- Preserve blocks listed in PREVIOUS VALIDATION RESULT.passed_blocks. Do not rename, reorder, shorten, or stylistically rewrite passed blocks unless a blocking issue explicitly depends on them.
- Keep stable ids/anchors and existing valid content where possible.
- If validator feedback conflicts with source materials, follow the source materials and explain the choice in agent_notes.
- If SOURCE CONTRACT FROM JSON contains authoritative_task_ids, never add or preserve task ids outside that list. Ignore validator feedback that asks for non-authoritative tasks and explain this in agent_notes.

STRUCTURED OUTPUT CONTRACT:
Fill the GeneratedMaterial structured output fields:
- content: the final user-facing material as HTML only, starting with <style> and containing <div class="cc-lesson">...</div>
- agent_notes: short notes about used inputs

The value of content is the final user-facing material. It must contain HTML only:
- it starts with <style>
- it ends at the final closing </div>
- it must not contain this JSON schema, validation instructions, markdown fences, "Return only JSON", or any text after the final </div>
- on retry, reuse valid previous content and edit only the blocks that require repair
""".strip()


def build_generator_system_prompt(spec: MaterialSpec) -> str:
    return (
        f"You are {spec.agent_type}. Use only the provided JSON fields, existing prompt/skill files, and Markdown references. "
        "Do not use external sources. Return the configured structured output fields. "
        "The content field must be HTML only and must not include runtime or validator instructions."
    )


def build_validation_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    content: str,
    rule_result: ValidationResult,
) -> str:
    return f"""
Проверь один материал. Не исправляй и не перегенерируй контент.

Fill MaterialValidationDecision structured output fields with this meaning:
{{
  "approved": true,
  "issues": [],
  "fix_instructions": [],
  "issues_by_block": [
    {{
      "block_id": "#selfcheck",
      "block_heading": "Проверка себя",
      "severity": "blocking",
      "issue": "what is wrong in this block",
      "fix_instruction": "what to change in this block"
    }}
  ],
  "passed_blocks": [
    {{
      "block_id": "#concepts",
      "block_heading": "Ключевые понятия",
      "reason": "why this block needs no changes"
    }}
  ]
}}
If RULE VALIDATOR ISSUES is not empty, approved must be false.

VALIDATION RESPONSIBILITY:
The deterministic rule validator checks only technical invariants: response boundary, allowed HTML container, dangerous tags, service marker leakage, source locator leakage, package statuses and dependency order.
You are responsible for semantic validation. Evaluate the material against the JSON, prompt/skill files, Markdown references, material kind, dependencies, and validation issues.
Check semantic completeness, internal consistency, source faithfulness, sufficient depth for the configured academic hours, suitability for the target learner/teacher, required task/assessment composition, and absence of contradictions or invented facts.
Do not reject a material only because section headings use different wording, unless a prompt/skill/reference explicitly requires exact fixed headings. Judge whether the required meaning and content are present.

MATERIAL-KIND VALIDATION POLICY:
{validation_policy_for_spec(spec)}

{channel_key_visibility_policy_for_spec(spec)}

BLOCK-LEVEL REPORTING:
- For every blocking issue, add a corresponding object to issues_by_block.
- Use the nearest HTML heading id as block_id, for example "#selfcheck"; if there is no id, use the heading text.
- Do not put a block into passed_blocks if it has any issue that requires editing.
- Add blocks to passed_blocks when they are semantically valid and should be preserved during retry.
- Keep fix_instructions actionable and localized to the named block when possible.

AGENT TYPE:
MaterialValidatorAgent

CHECKED MATERIAL KIND:
{spec.kind}

CHECKED MATERIAL TYPE:
{spec.material_type}

PROMPT/SKILL FILES FOR CHECKED MATERIAL:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}

CHECKED MATERIAL HTML START
{content}
CHECKED MATERIAL HTML END

Only text between CHECKED MATERIAL HTML START and CHECKED MATERIAL HTML END is the checked material.
Do not treat the JSON response schema above as material content.
""".strip()


def build_validator_system_prompt() -> str:
    return (
        "You are MaterialValidatorAgent. Check only the material between explicit material delimiters. "
        "Do not generate or repair content. Return the configured structured validation fields."
    )


def build_validation_controller_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    content: str,
    rule_result: ValidationResult,
    llm_result: ValidationResult,
    merged_validation: ValidationResult,
) -> str:
    return f"""
Review a material after all configured generation/validation attempts were exhausted.
Do not generate, rewrite, or repair the material. Decide whether the validator rejection should remain blocking.

Fill ValidationControllerDecision structured output fields with this meaning:
{{
  "approved": false,
  "decision": "approve_material_or_keep_failed",
  "quality_score": 0,
  "score_rationale": "why this score was assigned",
  "rationale": "short explanation",
  "blocking_issues": [],
  "non_blocking_issues": [],
  "overruled_validator_issues": [],
  "residual_risks": [],
  "fix_instructions": []
}}

ROLE:
You are ValidationControllerAgent, an independent quality controller for the generation pipeline.
Your job is to audit the validator decision, not to judge the generator effort or retry history.
Treat the validator report as the object under review and the material as evidence for checking whether the validator interpreted requirements proportionally.
Do not independently hunt for new defects that the validator did not report. If you notice a new minor issue, list it as residual risk, not as a new blocking issue.
Default stance: be skeptical of validator strictness. The validator bears the burden of proving that an issue is genuinely blocking. If the artifact is usable for its educational purpose and the validator objection is debatable, editorial, or based on a narrower reading than the sources require, prefer approving the material and recording residual risks.

APPELLATE REVIEW METHOD:
- Treat the validator's factual claims, severity labels, and interpretation of sources as unproven until you verify them against the checked material, JSON, prompt/skill files, and references.
- For each validator issue, ask: is the claimed defect actually present, does it contradict a source or core requirement, and would it materially prevent the artifact from serving its learner/teacher purpose?
- Overrule a validator issue when the claim is unsupported by the checked material, based on a debatable interpretation, based on optional style, or fixable by a narrow local edit without changing the material's source meaning.
- Do not let one local wording or terminology imperfection fail the whole material unless that imperfection teaches a materially wrong core rule, breaks required task composition, reveals forbidden answers, or makes the artifact unsafe/unusable.
- Distinguish educational simplification from factual error. A simplified beginner-facing statement is acceptable when it is pedagogically useful, does not contradict the course sources, and does not lead the learner to solve tasks incorrectly.
- Distinguish recommended course convention from language/platform specification. A convention can be stated as the course's recommended practice without being treated as a false claim about all possible syntax.
- If the validator found a real but local problem, prefer approve_material with non_blocking_issues and fix_instructions when the rest of the material is usable and the fix does not require regenerating the whole artifact.

DECISION POLICY:
- Never approve if deterministic technical checks failed, if content is empty, unsafe, malformed, outside the requested material kind, or leaks forbidden service/source markers.
- Keep failed if there are major source contradictions, invented facts that change the task, missing core content, wrong audience, insufficient depth for configured hours, missing required task/assessment composition, or unresolved dependency problems.
- You may approve when remaining objections are editorial preferences, harmless wording choices, overly literal interpretation, optional/recommended style, non-blocking section naming differences, unproven "potential" concerns, or minor imperfections that do not harm the artifact purpose.
- If SOURCE CONTRACT FROM JSON contains authoritative_task_ids, validator objections about missing keys/content for task ids outside that list are invalid and should be overruled. Do not let non-authoritative task ids block an otherwise usable material.
- You may approve when the material has a small number of local factual/terminology imprecisions that are easy to correct and do not undermine the central teaching objective; record them as non_blocking_issues with precise fix_instructions.
- Keep failed for factual errors only when they are central, repeated, source-contradicting, or likely to make the learner form a wrong mental model that affects task performance. Do not keep failed for a single localized nuance if the surrounding explanation is still usable.
- For practice tasks that are underspecified for deterministic stdout, prefer approving when the material preserves the source task, avoids fake values/expected output, and clearly marks tests as absent/not applicable/manual/unavailable until clarification. Do not keep failed merely because the material uses wording such as "training task", "manual check", "without autocheck", or "until source clarification".
- For practice wording, do not treat obvious restatements of the source as inventions. If the source lists two requested entities, "two variables" is a permissible restatement unless the material invents concrete names, concrete values, or exact output format.
- For theory boundary issues, require the validator to prove copyable answer leakage, not just conceptual similarity. Conceptual overlap with practice skills is expected and allowed in theory.
- A theory example is blocking only when it is a complete solved answer to a specific practice task and materially reuses source-specific variables, literals, target phrase, exact expected output, or the full task structure from that practice task.
- Generic lesson concepts and neutral examples are not answer leakage: print("text"), print(variable), print("label:", value), print with comma-separated arguments, text-vs-variable comparisons, and self-check answers may be necessary explanations.
- Self-check answers are allowed when they explain concepts. Treat them as blocking only when they contain a full concrete solution to a specific practice task.
- If the validator's issue is framed as "could be used as a template", "similar to the practice skill", or "same general operation" without a concrete match to source-specific practice task data, treat that issue as over-strict and non-blocking.
- Treat prompt/skill files, JSON and Markdown references as sources with possible priority conflicts. Explain how you resolved conflicts.
- Do not create new requirements. Do not enforce a narrower interpretation than the sources support.
- If approved, list any residual non-blocking risks. If not approved, list only genuinely blocking issues.

QUALITY SCORE:
- 0: unusable or unsafe material; technical/rule failure or wrong artifact kind.
- 1: severe source contradiction, missing core content, or answer leakage that defeats the artifact purpose.
- 2: significant quality problems remain; usable only after material edits.
- 3: usable for the requested purpose; remaining issues are limited, local, or non-blocking.
- 4: good material with minor residual risks or editorial improvements.
- 5: publication-ready material with no meaningful residual risk.
Assign the score after discounting validator objections that you overrule as disproportionate. Do not assign 0-2 solely because the validator found an issue; assign 0-2 only when the validator's blocking interpretation is actually justified.
When only local/editorial cleanup remains and the material can be used without changing the source meaning, the score should be at least 3.

MATERIAL-KIND VALIDATION POLICY:
{validation_policy_for_spec(spec)}

{channel_key_visibility_policy_for_spec(spec)}

CHECKED MATERIAL KIND:
{spec.kind}

CHECKED MATERIAL TYPE:
{spec.material_type}

PROMPT/SKILL FILES FOR CHECKED MATERIAL:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

RULE VALIDATION:
{compact_json(validation_result_summary(rule_result))}

LLM VALIDATION:
{compact_json(validation_result_summary(llm_result))}

MERGED VALIDATION:
{compact_json(validation_result_summary(merged_validation))}

CHECKED MATERIAL HTML START
{content}
CHECKED MATERIAL HTML END
""".strip()


def build_validation_controller_system_prompt() -> str:
    return (
        "You are ValidationControllerAgent. You review whether validation failures are genuinely blocking. "
        "You do not generate or repair content. Return the configured structured controller fields."
    )


def build_package_validation_prompt(
    *,
    task: dict[str, Any],
    specs: list[MaterialSpec],
    materials: list[MaterialResult],
    rule_result: ValidationResult,
) -> str:
    return f"""
Проверь полный пакет материалов. Не перегенерируй контент.
Package validation is advisory: report risks and warnings, but do not use this step to stop publication when all individual MaterialResult objects already have status="approved".

Fill PackageValidationDecision structured output fields with this meaning:
{{
  "approved": true,
  "issues": [],
  "fix_instructions": []
}}

PACKAGE VALIDATION JUDGEMENT POLICY:
- You receive full final HTML content for every generated material in FULL MATERIALRESULT OBJECTS. The content is not a preview and is not intentionally truncated.
- Do not claim that a material is truncated unless content_truncated is true or html_diagnostics proves an actual broken boundary in full_final_content.
- Validate only the current task package and the expected material specs listed below. Do not fail this lesson package because of future lessons, module-wide rows, or assessment content that was not generated in this task.
- Source JSON may contain known warnings such as difficulty.violation or module totals.raw percentages. If generated materials faithfully preserve and disclose such source warnings, report them as source_data_warnings, not as package-blocking material defects.
- Do not ask the generator to change lesson.practice_tasks counts or L1/L2 distribution when those values come from the source JSON. The generator must not silently rewrite source task composition to satisfy a norm.
- Approved individual materials remain approved at package validation time unless deterministic package rule issues prove missing materials, wrong order, or broken final files.
- Prefer approved=true with advisory issues when the materials are complete, importable HTML and individual material validation already approved them.

TASK JSON:
{compact_json(task)}

EXPECTED MATERIAL SPECS:
{compact_json([{"kind": spec.kind, "type": spec.material_type, "agent": spec.agent_type} for spec in specs])}

FULL MATERIALRESULT OBJECTS:
{compact_json([package_material_payload(item) for item in materials])}

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}
""".strip()
