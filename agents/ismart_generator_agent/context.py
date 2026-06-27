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
                "When approved practice dependency contains generation_artifacts.practice_instances, use those exact instances, tests, hidden_solution, and teacher_explanation as the source of truth.",
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

    if spec.kind == "mr_intermediate":
        return {
            "contract_type": "mr_intermediate_guidance_contract",
            "dependency_source": "approved intermediate MaterialResult with generation_artifacts.intermediate_assessment",
            "generation_rules": [
                "Create publishable teacher methodical guidance for conducting and checking the approved intermediate assessment.",
                "Use dependency intermediate.generation_artifacts.intermediate_assessment as the source of internal keys, reference answers, rubrics, tests, and hidden solutions.",
                "Do not reconstruct assessment variants, tasks, keys, tests, rubrics, or hidden solutions from scratch.",
                "Do not duplicate the full variant-by-variant answer bank, reference answers, code solutions, stdin/stdout tests, or autocheck configs in mr_intermediate HTML.",
                "The HTML may explain how the teacher should use the internal assessment/QA artifact, how to conduct the 45+45 minute assessment, how to apply criteria, and how to react to typical mistakes.",
                "Limited generic examples of checking logic are allowed only when they do not reveal answers to the current generated variants.",
            ],
            "validation_rules": [
                "Do not require a full key bank inside mr_intermediate HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment.",
                "Reject if mr_intermediate HTML duplicates full keys, reference answers, code solutions, stdin/stdout tests, or autocheck configs from intermediate_assessment.",
                "Reject if mr_intermediate lacks practical methodical guidance for conducting, checking, scoring, and responding to typical mistakes.",
            ],
        }

    if spec.kind == "specification_qa" and normalized_tasks:
        lesson = task.get("lesson") or {}
        authoritative_task_ids = [item["id"] for item in normalized_tasks]
        return {
            "contract_type": "specification_qa_practice_task_contract",
            "required_task_count": len(normalized_tasks),
            "authoritative_task_ids": authoritative_task_ids,
            "hours_practice": (lesson.get("hours") or {}).get("practice"),
            "difficulty": lesson.get("difficulty") or {},
            "tasks": normalized_tasks,
            "generation_rules": [
                "Build QA/specification only for tasks listed in authoritative_task_ids, in the same order and with the same P id.",
                "When approved practice dependency contains generation_artifacts.practice_instances, use those exact instances, tests, hidden_solution, and teacher_explanation as the source of truth.",
                "Preserve each task pattern. Do not invent concrete variable names, concrete values, exact stdout, exact input data, or a mandatory output format unless they are explicitly present in source_text, Markdown references, or approved dependency artifacts.",
                "If a task is underspecified for deterministic stdout, mark it as requiring source clarification or manual checking. Do not create deterministic autocheck tests, mandatory expected output, or mandatory reference code with invented values.",
                "For underspecified tasks, an optional illustrative teacher example is allowed only if clearly labeled as a non-authoritative example and not used as the required test/key.",
                "Reuse the approved practice and mr_practice dependency interpretation when they mark a task as underspecified or manually checked.",
                "Do not claim full JSON conformance if QA introduced values, tests, or formats that are not in the source.",
            ],
            "validation_rules": [
                "Approve QA/specification only when task ids match authoritative_task_ids exactly.",
                "Reject deterministic tests or keys for underspecified source tasks when they rely on invented concrete values.",
                "Do not reject merely because an underspecified task has no deterministic test; that is the correct source-faithful representation.",
            ],
        }

    if spec.kind == "self_work":
        lesson = task.get("lesson") or {}
        return {
            "contract_type": "self_work_autocheck_contract",
            "hours_self_study": (lesson.get("hours") or {}).get("self_study"),
            "difficulty": lesson.get("difficulty") or {},
            "practice_tasks": normalized_tasks,
            "required_independent_task_count": 8,
            "required_selfcheck_question_count": 10,
            "generation_rules": [
                "Build internal checking artifacts before learner-facing HTML is rendered.",
                "Create exactly 8 independent self-work task checks unless the source JSON explicitly requires another count.",
                "Create exactly 10 self-check question configs unless the source JSON explicitly requires another count.",
                "For every independent task, provide a checked skill and either a correct_answer, runtime_tests, or manual_check_rules.",
                "For every self-check question, provide template_code, student_prompt, correct_answers, and autocheck_config.",
                "Keep correct answers, autocheck config, internal explanations, and platform keys inside generation_artifacts.self_work_autocheck only.",
                "The learner-facing self_work HTML must render clear tasks/questions but must not display correct answers, answer flags, filled templates, or internal platform config.",
                "Use template_descriptions to choose platform-compatible question/template codes such as question, 6A, 6D, 6G, 8D, 10D, or 3H when appropriate.",
            ],
            "validation_rules": [
                "Do not require visible keys in learner-facing self_work HTML when generation_artifacts.self_work_autocheck contains the internal keys/config.",
                "Reject if internal self_work_autocheck artifacts are missing, have no correct answers/config, or contradict the learner-facing questions.",
                "Reject if the learner-facing HTML displays correct answers or internal autocheck configuration.",
            ],
        }

    if spec.kind == "intermediate":
        module = task.get("module") or {}
        return {
            "contract_type": "intermediate_assessment_contract",
            "required_variant_count": 4,
            "required_closed_questions_per_variant": 16,
            "required_open_questions_per_variant": 4,
            "required_code_tasks_per_variant": 3,
            "module_lessons": module.get("lessons") or [],
            "generation_rules": [
                "Build internal assessment artifacts before learner-facing HTML is rendered.",
                "Use the module JSON and Markdown references as the source of module topics and skill coverage.",
                "For intermediate assessment, the assessment composition is governed by the intermediate MaterialSpec/skill: 4 variants, each with 16 closed questions, 4 open questions, and 3 code tasks.",
                "If the attestation lesson JSON gives a smaller summary count, treat it as source context for topics and timing, not as permission to omit the MaterialSpec assessment composition.",
                "For every closed question, provide template_code, student_prompt, correct_answers, and autocheck_config.",
                "For every open question, provide student_prompt, reference_answer, and rubric.",
                "For every code task, provide student_condition, input/output requirements, hidden_solution, and runtime_tests or manual_check_rules.",
                "Keep correct answers, reference answers, rubrics, hidden solutions, and autocheck config inside generation_artifacts.intermediate_assessment only.",
                "The learner-facing intermediate HTML must render only tasks/questions/conditions and must not display keys, эталоны, hidden solutions, or internal platform config.",
            ],
            "validation_rules": [
                "Do not require visible keys in learner-facing intermediate HTML when generation_artifacts.intermediate_assessment contains the internal keys/config.",
                "Reject if intermediate_assessment artifacts are missing, structurally incomplete, lack keys/tests/rubrics, or contradict the learner-facing assessment.",
                "Reject if learner-facing HTML displays keys, эталоны, reference answers, hidden solutions, or internal autocheck configuration.",
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
            "Generate exactly one student-facing task variant for each task in tasks, in the same order and with the same P id.",
            "Treat source_text as an authoritative task pattern: preserve id, level, task_type, skill_target, and constraints, but do not treat source_text as mandatory final learner wording unless it explicitly says exact wording/value is required.",
            "Create a new concrete variant of the same pattern: use different scenario, variable names, literals, input data, expected output, and code shape from theory, Markdown references, and dependency materials unless the source explicitly requires exact values.",
            "Do not replace the task pattern with a different skill or task type.",
            "Expected output is not invented when it is deterministically derived from explicit literals, assignments, and exact print(...) calls in source_text using standard Python print semantics.",
            "For each task, visibly include learner-facing structured fields required by the practice references: level, condition, input data, output requirement, how to check, and tests/status. Do not show source_text, source task, source contract, or JSON/pipeline wording in learner-facing practice.",
            "Tests must be explicit input -> expected output pairs when the source task defines enough information for deterministic stdout.",
            "If the source task is underspecified for deterministic stdout, preserve the same task structure and state that the exact autocheck test needs source clarification instead of inventing fake expected output.",
            "For an underspecified task, do not render a fake input -> expected output table or placeholder expected output; render a tests/status block that says tests are absent/not applicable until source clarification.",
            "Do not show corrected code, answers, keys, rc, stderr, or stdout logs in the student practice material.",
        ],
        "pipeline": [
            "PracticeTaskTemplateAgent extracts PracticeTaskTemplate objects from source tasks.",
            "PracticeTaskVariantAgent fills templates into PracticeTaskInstance objects with new concrete values.",
            "PracticeMaterialAgent renders only student-facing fields from PracticeTaskInstanceSet into HTML.",
        ],
    }


def channel_key_visibility_policy_for_spec(spec: MaterialSpec) -> str:
    return f"""
CHANNEL AND KEY VISIBILITY POLICY:
- Resolve all answer-key and solution visibility requirements by artifact channel, not as a single global rule.
- Learner-facing lesson materials are theory, practice, and self_work. They must not expose lesson practice solution keys, corrected practice code, QA ids, source hashes, rc/stderr/stdout logs, internal verification traces, internal Markdown paths, source locators, local filenames, or working-folder references such as docs/..., .md, референсы, рабочая область агента.
- Self-check/autocheck keys for learner-facing self_work must not be visible in HTML. If the platform needs keys, treat them as non-rendered/internal platform configuration, not as displayed text. Do not show learner-facing blocks such as "Ключ", "Правильный вариант", "{{%answer%}}", or filled input templates with expected answers.
- For self_work, internal platform keys and autocheck settings should be carried in generation_artifacts.self_work_autocheck. They are evidence for validation and QA, not learner-facing content.
- For intermediate, assessment keys, reference answers, rubrics, code solutions, tests, and autocheck settings should be carried in generation_artifacts.intermediate_assessment. They are evidence for validation and QA, not learner-facing content.
- Teacher-facing artifacts are mr_theory, mr_practice, and mr_intermediate. Teacher guidance may include answers, solution keys, code snippets, rubrics, and checking guidance only when the material-specific policy permits them.
- mr_theory is teacher-facing and must be judged as teacher guidance for the theory lesson, not as a duplicate student theory material.
- mr_practice is teacher-facing and is expected to include keys/solutions for all practice tasks. Do not reject mr_practice merely because it includes teacher keys. Reject mr_practice if required teacher keys are absent, contradict the source/practice material, or leak QA/internal ids, source hashes, or service locators.
- mr_intermediate is teacher-facing publishable methodical guidance. It may explain how to conduct and check the assessment, but it must not duplicate the full variant-by-variant key bank, reference answers, hidden code solutions, stdin/stdout tests, or autocheck configs in HTML when generation_artifacts.intermediate_assessment is available.
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

    if spec.kind == "mr_intermediate":
        return """
MR_INTERMEDIATE VALIDATION POLICY:
- mr_intermediate is teacher-facing publishable methodical guidance for conducting and checking the approved intermediate assessment.
- The source of truth for full keys, reference answers, rubrics, code solutions, runtime tests, and autocheck configs is dependency intermediate.generation_artifacts.intermediate_assessment.
- Do not require a full key bank, full reference-answer bank, full code solutions, stdin/stdout tests, or autocheck configs inside mr_intermediate HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment.
- Reject mr_intermediate HTML if it duplicates full variant-by-variant keys, reference answers, hidden code solutions, stdin/stdout tests, or autocheck configs from the intermediate_assessment artifact.
- Approve methodical guidance that explains how to conduct the assessment, how to use the internal QA/artifact layer, how to score closed/open/code tasks at a general level, how to handle appeals/typical errors, and how to keep answer keys out of student access.
- Do not reject merely because the document says keys/tests are available in the internal QA/artifact layer; that is the expected boundary for intermediate assessment.
- Reject if the document is only a pointer to the artifact and lacks useful teacher guidance for preparation, timing, checking workflow, scoring policy, and typical teacher responses.
""".strip()

    if spec.kind == "specification_qa":
        return """
SPECIFICATION_QA VALIDATION POLICY:
- specification_qa is QA/internal-facing. It may contain QA ids, source hashes, traceability, keys, tests, faulty/fixed code, and validation data.
- Validate task ids against SOURCE CONTRACT FROM JSON.authoritative_task_ids when present. Do not infer extra tasks from examples or module-wide context.
- When approved practice dependency contains generation_artifacts.practice_instances, validate QA against those exact instances, tests, hidden_solution, and teacher_explanation.
- For each practice task, preserve the source pattern and approved practice instance meaning. Do not require or approve invented concrete values, variable names, exact stdout, exact stdin, or mandatory output format unless they are explicit in source_text, references, or approved dependency artifacts.
- If a source task is underspecified for deterministic stdout, QA should mark deterministic autocheck as unavailable/needs source clarification/manual check. Do not reject QA merely because such a task has no deterministic test.
- For underspecified tasks, optional example code is acceptable only when clearly labeled as non-authoritative and not used as the expected output, key, or platform test.
- If approved practice or mr_practice dependency marks a task as underspecified/manual, reuse that interpretation instead of creating deterministic tests.
- Reject if QA claims "JSON conformance" while adding invented values/tests/formats, or if keys/tests/faulty patches contradict the approved materials.
""".strip()

    if spec.kind == "self_work":
        return """
SELF_WORK VALIDATION POLICY:
- self_work is learner-facing. It must include independent work tasks and self-check questions, but visible answer keys are forbidden in the published HTML.
- Do not require visible keys in self_work HTML. If platform autocheck keys are needed, they belong to a non-rendered/internal platform layer or QA/internal artifact, not to learner-facing text.
- Treat VALIDATION VIEW OF GENERATION ARTIFACTS.self_work_autocheck as the internal platform/QA layer. It may contain correct_answers, runtime_tests, autocheck_config, and internal_explanation.
- When self_work_autocheck contains complete internal keys/config for the visible tasks/questions, do not reject the HTML merely because the keys are not visible.
- Reject if self_work_autocheck is missing, lacks correct answers/config for generated self-check questions, or contradicts the learner-facing questions/tasks.
- Reject visible answer disclosures such as "Ключ", "Правильный вариант", explicit answer lists, {%answer%} blocks, filled {{input-text:answer}} templates, or highlighted correct options.
- Approve self-check questions without visible keys when the learner-facing task text is clear and no answers are displayed.
""".strip()

    if spec.kind == "intermediate":
        return """
INTERMEDIATE VALIDATION POLICY:
- intermediate is learner-facing assessment HTML. It must include the visible assessment tasks, but visible answer keys, reference answers, rubrics with exact answers, hidden solutions, and autocheck configs are forbidden in the published HTML.
- Treat VALIDATION VIEW OF GENERATION ARTIFACTS.intermediate_assessment as the internal assessment/QA layer. It may contain correct_answers, reference_answer, rubric, runtime_tests, hidden_solution, teacher_explanation, autocheck_config, and internal_explanation.
- When intermediate_assessment contains complete internal keys/config for the visible variants, do not reject the HTML merely because keys/эталоны/tests are not visible.
- Validate the structure against SOURCE CONTRACT FROM JSON: 4 variants, each with 16 closed questions, 4 open questions, and 3 code tasks, unless the source contract is explicitly changed.
- Reject if intermediate_assessment is missing, lacks internal keys/rubrics/tests for generated tasks, or contradicts the learner-facing assessment.
- Reject visible answer disclosures such as "Ключ", "Ключи", "Эталон", "Эталоны", "Правильный ответ", explicit answer lists, hidden_solution, reference_answer, correct_answers, or autocheck_config inside the checked HTML.
- Candidate answer options in closed questions are allowed in learner-facing HTML. They are not answer-key leakage unless the correct option is explicitly marked or a key/answer section is printed.
- A 10D/6A/6D/6G/8D label in learner-facing HTML is a visible task-type label only. Do not require platform-import markup such as {{input-text:...}} in publishable HTML when intermediate_assessment carries the structured template_code and autocheck_config.
- "Find/fix/explain the error" assessment tasks are allowed when the learner-facing HTML shows only the faulty code or condition. Treat them as leakage only when the corrected/reference solution is visible in HTML.
- Do not treat the attestation lesson JSON summary count as overriding skill 05 composition. Use it for topic/timing context unless the source contract says otherwise.
""".strip()

    if spec.kind != "practice":
        return "No additional material-kind validation policy."

    return """
PRACTICE VALIDATION POLICY:
- Practice generation is template-based. lesson.practice_tasks defines the authoritative pattern: P id, level, task type, target skill, and constraints. It is not necessarily the final learner wording.
- Approve a new task variant when it preserves the source pattern but uses different scenario, variable names, literals, input/output data, and code shape from theory, Markdown references, and dependency materials.
- Reject direct copying from theory/references/dependencies when the copied content becomes the learner-facing practice task, starter code, test data, or expected output without an explicit source requirement.
- Do not require the practice task to preserve source_text verbatim. Require pattern faithfulness, not textual identity.
- Practice references require tests as visible input -> expected output pairs when the source task provides enough information for deterministic stdout.
- Visible expected stdout in a student-facing deterministic test table is allowed and expected for practice materials. It is a test oracle, not a forbidden answer key, as long as it does not include corrected code, hidden_solution text, teacher_explanation, QA ids, source hashes, or internal trace data.
- Do not reject practice merely because deterministic tests show concrete expected stdout. Reject only when the material reveals the corrected code, tells the exact edit operation, exposes hidden_solution/teacher_explanation, or invents unsupported expected output.
- For refactoring tasks, separate runtime tests from manual/static checks. Runtime tests verify behavior preservation through stdin -> expected stdout. Manual/static checks verify requirements that stdout cannot prove: better variable names, comments, removed duplicated computation, named constants, and code readability.
- Do not reject a refactoring practice merely because stdout tests cannot prove variable names, comments, or duplication removal when the learner-facing material includes explicit manual/static checks for those requirements.
- Reject a refactoring practice when it has neither runtime tests for behavior that is actually runnable nor manual/static checks for non-runtime requirements.
- Do not require source_text, source task, source contract, JSON wording, or pipeline trace to be visible in learner-facing HTML. Those are internal artifacts for validation and MR/QA. Their absence from student practice is correct.
- Do not treat an expected output as invented when it follows deterministically from explicit source_text literals, assignments, and exact print(...) calls using standard Python print semantics.
- For source tasks that are underspecified for deterministic stdout, do not require fake values or fake expected output. The material may mark tests as absent/not applicable until source clarification, but it must preserve the same visible learner-facing task structure: level, condition, input data, output requirement, how to check, and tests/status.
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
        "generation_artifacts": material.generation_artifacts,
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
        "generation_artifacts": material.generation_artifacts,
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


def _sanitize_practice_instances_for_validation(instances: Any) -> Any:
    if not isinstance(instances, dict):
        return instances
    sanitized: dict[str, Any] = {key: value for key, value in instances.items() if key != "tasks"}
    tasks = instances.get("tasks")
    if not isinstance(tasks, list):
        return sanitized
    sanitized_tasks: list[Any] = []
    for item in tasks:
        if not isinstance(item, dict):
            sanitized_tasks.append(item)
            continue
        sanitized_tasks.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"hidden_solution", "teacher_explanation"}
            }
        )
    sanitized["tasks"] = sanitized_tasks
    return sanitized


def sanitize_generation_artifacts_for_validation(
    spec: MaterialSpec,
    generation_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the artifact view safe for learner-facing validation prompts."""
    if not generation_artifacts:
        return {}
    if spec.kind in {"mr_practice", "specification_qa"}:
        return dict(generation_artifacts)

    sanitized: dict[str, Any] = {}
    for key, value in generation_artifacts.items():
        if key == "practice_instances":
            sanitized[key] = _sanitize_practice_instances_for_validation(value)
        else:
            sanitized[key] = value
    if "practice_instances" in generation_artifacts:
        sanitized["artifact_visibility_note"] = (
            "This is a validation view of internal generation artifacts. Teacher-only solution/explanation "
            "fields are redacted here. Treat answer leakage as present only when the checked HTML or the "
            "deterministic rule report contains the leaked learner-facing text."
        )
    if "self_work_autocheck" in generation_artifacts:
        sanitized["self_work_autocheck_visibility_note"] = (
            "This self_work_autocheck object is an internal platform/QA layer. Its correct_answers, "
            "runtime_tests, autocheck_config, and internal_explanation fields are allowed here and must not "
            "be treated as learner-facing leakage unless the checked HTML displays them."
        )
    if "intermediate_assessment" in generation_artifacts:
        sanitized["intermediate_assessment_visibility_note"] = (
            "This intermediate_assessment object is an internal assessment/QA layer. Its correct_answers, "
            "reference_answer, rubric, runtime_tests, hidden_solution, teacher_explanation, autocheck_config, "
            "and internal_explanation fields are allowed here and must not be treated as learner-facing leakage "
            "unless the checked HTML displays them."
        )
    return sanitized


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
    generation_artifacts: dict[str, Any] | None = None,
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

GENERATION ARTIFACTS FOR THIS MATERIAL:
{compact_json(generation_artifacts or {})}

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


def build_practice_template_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    previous_artifacts: dict[str, Any] | None,
    previous_issues: list[str],
) -> str:
    return f"""
Build structured practice task templates. Do not generate final learner HTML.

AGENT TYPE:
PracticeTaskTemplateAgent

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

PREVIOUS PRACTICE GENERATION ARTIFACTS:
{compact_json(previous_artifacts or {})}

PREVIOUS VALIDATION/STRUCTURAL ISSUES:
{compact_json(previous_issues)}

TEMPLATE RULES:
- Return PracticeTaskTemplateSet structured output only.
- Create exactly one PracticeTaskTemplate for every SOURCE CONTRACT task, in the same order and with the same P id and level.
- Extract task_type, skill_target, invariants, slots_to_fill, constraints, and test_policy from source_text, JSON context, prompt/skill files, and Markdown references.
- Treat lesson.practice_tasks as a pattern source, not necessarily as final learner wording.
- Do not invent final scenario values, variable names, concrete inputs, concrete outputs, or final code unless the source explicitly requires exact values; put such needs into slots_to_fill/constraints/test_policy.
- Preserve exact values only when the source clearly requires exact values or exact wording.
""".strip()


def build_practice_variant_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    templates: dict[str, Any],
    previous_artifacts: dict[str, Any] | None,
    previous_issues: list[str],
) -> str:
    return f"""
Fill practice task templates into concrete task variants. Do not generate final learner HTML.

AGENT TYPE:
PracticeTaskVariantAgent

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

PRACTICE TASK TEMPLATES:
{compact_json(templates)}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

ANTI-COPY SOURCE MATERIALS:
{compact_json([material_result_summary(item) for item in dependencies])}

PREVIOUS PRACTICE GENERATION ARTIFACTS:
{compact_json(previous_artifacts or {})}

PREVIOUS VALIDATION/STRUCTURAL ISSUES:
{compact_json(previous_issues)}

VARIANT RULES:
- Return PracticeTaskInstanceSet structured output only.
- Create exactly one PracticeTaskInstance for every PracticeTaskTemplate, in the same order and with matching id/template_id.
- Preserve the template's level, task_type, skill_target, and invariants.
- Generate a new concrete scenario and new values: variable names, literals, input data, expected output, and code shape must differ from theory, Markdown references, and dependency materials unless the source explicitly requires exact values.
- Do not change the task into another skill or task type.
- Treat source_text/source task as internal pattern evidence. Never copy it into scenario, student_condition, input_requirements, or output_requirements, and never label learner-facing text as "source task", "from JSON", or "as in the lesson task".
- For retry, if previous issues mention source_text, source task, source contract, exact-fix hints, or student-facing fields revealing the fix, rewrite only the affected PracticeTaskInstance fields and keep valid task ids/order/tests where possible.
- For fix/debug tasks, the learner-facing fields scenario, student_condition, input_requirements, and output_requirements must not reveal the exact edit operation. Do not say "add a quote", "replace X with Y", "the quote is missing", "the function name is misspelled", or similar solution hints. The learner may see the faulty code, the error message when source-required, the expected behavior, and the test result target; the exact fix belongs only in hidden_solution and teacher_explanation.
- Put solutions, corrected code, answer keys, and teacher-only explanations only into hidden_solution and teacher_explanation. These fields are internal artifacts for MR/QA and must not be shown in learner HTML.
- If deterministic runtime checks are source-supported, put every visible stdin -> expected stdout check into runtime_tests and mirror the same list in legacy tests. Every item must use exactly the keys input and expected_output. Preserve trailing newlines in expected_output as "\\n"; do not use keys such as output/stdout/result.
- For refactoring tasks, use runtime_tests only to verify behavior preservation. Put requirements that stdout cannot prove into manual_checks: better variable names, comments, removed duplicated computation, named constants, and readability.
- Set run_mode to single_file when one runnable file is intended, separate_snippets when subtasks should be run independently, manual_only when only manual/static checking is meaningful, or needs_platform_clarification when the source does not define how the platform runs multi-part code.
- If deterministic runtime tests are not source-supported, keep runtime_tests and tests empty and state the manual/clarification policy in output_requirements, manual_checks, hidden_solution, and teacher_explanation instead of inventing fake exact outputs.
- In uniqueness_notes, explicitly state what differs from theory/references/dependencies.
""".strip()


def build_practice_template_system_prompt() -> str:
    return (
        "You are PracticeTaskTemplateAgent. Extract structured task templates from the provided JSON, "
        "prompt/skill files, references, and dependencies. Return only the configured structured output."
    )


def build_practice_variant_system_prompt() -> str:
    return (
        "You are PracticeTaskVariantAgent. Fill task templates into new concrete variants while preserving the "
        "source pattern and avoiding direct copying from theory/references/dependencies. Return only the configured structured output."
    )


def build_self_work_autocheck_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    previous_artifacts: dict[str, Any] | None,
    previous_issues: list[str],
) -> str:
    return f"""
Build the internal self-work checking/autocheck artifact. Do not generate learner-facing HTML.

AGENT TYPE:
SelfWorkAutocheckAgent

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

PREVIOUS SELF_WORK GENERATION ARTIFACTS:
{compact_json(previous_artifacts or {})}

PREVIOUS VALIDATION/STRUCTURAL ISSUES:
{compact_json(previous_issues)}

SELF-WORK ARTIFACT RULES:
- Return SelfWorkAutocheckSet structured output only.
- Create exactly 8 independent_tasks and exactly 10 selfcheck_questions unless the source JSON explicitly requires another count.
- independent_tasks are internal checking metadata for the independent work tasks. Each task must include id, student_task_title, checked_skill, checking_mode, and at least one of correct_answer, runtime_tests, or manual_check_rules.
- selfcheck_questions are internal platform/autocheck metadata for the self-check block. Each question must include id, template_code, question_type, skill_target, student_prompt, correct_answers, and autocheck_config.
- Use template_descriptions to choose platform-compatible template_code values. Prefer a varied mix when the references require multiple template kinds.
- Put correct answers, answer flags, matching pairs, ordering keys, underlining keys, runtime tests, and teacher-only explanations only in this internal artifact.
- Do not instruct the HTML generator to display correct_answers, answer flags, filled input templates, or internal_explanation.
- On retry, preserve valid previous questions/tasks where possible and repair only items connected to previous issues.
""".strip()


def build_self_work_autocheck_system_prompt() -> str:
    return (
        "You are SelfWorkAutocheckAgent. Build internal checking and autocheck configuration for self_work. "
        "Do not produce learner-facing HTML. Return only the configured structured output."
    )


def build_intermediate_assessment_artifact_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    previous_artifacts: dict[str, Any] | None,
    previous_issues: list[str],
) -> str:
    return f"""
Build the internal intermediate assessment artifact. Do not generate learner-facing HTML.

AGENT TYPE:
IntermediateAssessmentArtifactAgent

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

PREVIOUS INTERMEDIATE GENERATION ARTIFACTS:
{compact_json(previous_artifacts or {})}

PREVIOUS VALIDATION/STRUCTURAL ISSUES:
{compact_json(previous_issues)}

INTERMEDIATE ASSESSMENT ARTIFACT RULES:
- Return IntermediateAssessmentArtifact structured output only.
- Create exactly 4 variants.
- For each variant, create exactly 16 closed_questions, 4 open_questions, and 3 code_tasks.
- For every closed question, include id, template_code, skill_target, student_prompt, correct_answers, and autocheck_config.
- A closed question must have unambiguous grading. If more than one answer is defensible, either rewrite the prompt/options so exactly one answer is correct, or use a template/autocheck_config that supports multiple correct answers and list every correct answer. Never choose one arbitrary answer while internal_explanation says several answers are valid.
- Use a varied mix of platform-compatible template_code values. Include coded templates such as 6A, 6D, 6G, 8D, 10D where the references require template diversity.
- For every open question, include id, skill_target, student_prompt, reference_answer, and rubric.
- For every code task, include id, skill_target, student_condition, input_requirements, output_requirements, hidden_solution, and runtime_tests or manual_check_rules.
- Use module.lessons and Markdown references for topic coverage. Do not copy prior examples verbatim when a new assessment item can be created.
- Keep correct answers, reference answers, rubrics, hidden solutions, tests, and autocheck configs only in this internal artifact.
- The HTML renderer will use student_prompt/options/student_condition/input_requirements/output_requirements but must not display correct_answers, reference_answer, hidden_solution, teacher_explanation, or autocheck_config.
- On retry, preserve valid previous variants/items where possible and repair only items connected to previous issues.
""".strip()


def build_intermediate_assessment_artifact_system_prompt() -> str:
    return (
        "You are IntermediateAssessmentArtifactAgent. Build internal keys, rubrics, tests, and autocheck "
        "configuration for intermediate assessment. Do not produce learner-facing HTML. Return only the "
        "configured structured output."
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
    generation_artifacts: dict[str, Any] | None = None,
) -> str:
    validation_generation_artifacts = sanitize_generation_artifacts_for_validation(spec, generation_artifacts)
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

VALIDATION VIEW OF GENERATION ARTIFACTS FOR CHECKED MATERIAL:
{compact_json(validation_generation_artifacts)}

ARTIFACT VISIBILITY RULE:
- The section above is not learner-facing HTML. It is provided only to check task ids, generated scenarios, visible tests, manual/static checks, duplicate reports, and deterministic leak reports.
- For learner-facing materials, private teacher-only fields may be redacted from that section. Do not infer a learner-facing answer leak from internal artifacts. A leak is blocking only when the leaked content appears between CHECKED MATERIAL HTML START/END or RULE VALIDATOR ISSUES reports it.
- For self_work, self_work_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. If it is complete and consistent, it satisfies requirements for keys/autocheck without visible answer keys in HTML.
- For intermediate, intermediate_assessment is the required non-rendered assessment/QA layer for keys, эталоны, rubrics, tests, and solutions. If it is complete and consistent, it satisfies requirements for keys/autocheck without visible answer keys in HTML.

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
    generation_artifacts: dict[str, Any] | None = None,
) -> str:
    validation_generation_artifacts = sanitize_generation_artifacts_for_validation(spec, generation_artifacts)
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
- For practice materials, overrule validator objections that treat visible deterministic expected stdout as a forbidden key. Expected stdout in a test table is allowed when it is part of the learner's platform test description and does not reveal corrected code or the exact edit operation.
- For learner-facing practice materials, overrule validator objections that treat teacher-only fields inside internal generation artifacts as learner-facing leaks. A hidden solution/explanation leak is blocking only when the checked HTML contains the leaked content or RULE VALIDATION reports a deterministic learner-facing leak.
- For practice tasks that are underspecified for deterministic stdout, prefer approving when the material preserves the source task, avoids fake values/expected output, and clearly marks tests as absent/not applicable/manual/unavailable until clarification. Do not keep failed merely because the material uses wording such as "training task", "manual check", "without autocheck", or "until source clarification".
- For practice wording, do not treat obvious restatements of the source as inventions. If the source lists two requested entities, "two variables" is a permissible restatement unless the material invents concrete names, concrete values, or exact output format.
- For intermediate, overrule validator objections that treat candidate answer options as answer-key leakage. Closed questions must show answer options; this is blocking only when HTML explicitly marks the correct option or prints a key/answer section.
- For intermediate, overrule validator objections that require platform-import template markup such as {{input-text:...}} in publishable learner HTML when generation_artifacts.intermediate_assessment contains template_code and autocheck_config. The HTML is the readable student artifact; the internal artifact is the platform/autocheck layer.
- For intermediate, overrule validator objections that treat "find/fix/explain the error" prompts as leaked solutions merely because the error is local or obvious. Keep failed only when the corrected/reference solution itself is visible in HTML.
- For mr_intermediate, overrule validator objections that require the full intermediate key bank, reference answers, code solutions, stdin/stdout tests, or autocheck configs to be duplicated in HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment. The expected HTML is methodical guidance plus instructions for using the internal artifact, not the full answer bank.
- For mr_intermediate, keep failed when the HTML duplicates the full answer bank/tests/solutions from intermediate_assessment or lacks practical methodical guidance.
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

VALIDATION VIEW OF GENERATION ARTIFACTS FOR CHECKED MATERIAL:
{compact_json(validation_generation_artifacts)}

ARTIFACT VISIBILITY RULE:
- The section above is not learner-facing HTML. It is provided only to audit validator claims about task ids, generated scenarios, visible tests, manual/static checks, duplicate reports, and deterministic leak reports.
- For learner-facing materials, private teacher-only fields may be redacted from that section. Do not uphold a validator claim of learner-facing key leakage merely because an internal artifact contains teacher-only data. Uphold it only if the checked HTML contains the leaked content or RULE VALIDATION reports a deterministic learner-facing leak.
- For self_work, self_work_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. Validator objections about missing visible keys should be overruled when this artifact is complete and consistent with the HTML.
- For intermediate, intermediate_assessment is the required non-rendered assessment/QA layer for keys, эталоны, rubrics, tests, and solutions. Validator objections about missing visible keys should be overruled when this artifact is complete and consistent with the HTML.

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
