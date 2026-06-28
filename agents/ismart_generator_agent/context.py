from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import MaterialResult, MaterialSpec, ReferenceBundle, ValidationResult
from .sources import compact_json


def select_references(spec: MaterialSpec, references: ReferenceBundle) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for field in spec.reference_fields:
        selected[field] = [
            reference_document_for_prompt(document)
            for document in references.get(field, [])
        ]
    return selected


def reference_document_for_prompt(document: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "field": document.field,
        "document_name": Path(str(document.path)).stem,
        "truncated": document.truncated,
        "content": document.content,
    }
    return data


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
            "required_assessment_composition": {
                "variant_count": 4,
                "items_per_variant": 15,
                "test_questions_per_variant": 5,
                "open_code_questions_per_variant": 5,
                "practical_code_tasks_per_variant": 5,
                "minimum_code_writing_items_per_variant": 10,
            },
            "scoring_policy": {
                "allow_numeric_points_thresholds_or_grade_conversion": "only when exact numbers are explicitly present in JSON or Markdown references",
                "fallback_when_source_has_no_numeric_scale": "describe checking workflow, criteria groups, and local-organization result fixation without invented points, pass thresholds, or grade conversion tables",
            },
            "generation_rules": [
                "Create publishable teacher methodical guidance for conducting and checking the approved intermediate assessment.",
                "Use dependency intermediate.generation_artifacts.intermediate_assessment as the source of internal keys, reference answers, rubrics, tests, and hidden solutions.",
                "Do not reconstruct assessment variants, tasks, keys, tests, rubrics, or hidden solutions from scratch.",
                "State the approved assessment composition explicitly: 4 variants; each variant has 15 items: 5 test questions, 5 open-code questions, and 5 practical code tasks; at least 10 of 15 items require writing code.",
                "Do not duplicate the full variant-by-variant answer bank, reference answers, code solutions, stdin/stdout tests, or autocheck configs in mr_intermediate HTML.",
                "Do not print literal internal field names such as intermediate_assessment, generation_artifacts, hidden_solution, or autocheck_config in publishable mr_intermediate HTML; use neutral wording such as closed teacher checking layer.",
                "The HTML may explain how the teacher should use the internal assessment/QA artifact, how to conduct the 45+45 minute assessment, how to apply criteria, and how to react to typical mistakes.",
                "Do not invent a numeric scoring scale, maximum points, pass/fail threshold, percent boundary, or conversion to a 5-point grade. Use numeric scoring only when those exact numbers are present in JSON or Markdown references.",
                "If sources do not define a numeric scale, describe checking by criteria groups and closed teacher checking layer, and say that final result fixation follows the organization's local rules without naming thresholds.",
                "On retry, if validator reports unsupported scoring numbers or grade conversion, remove those numbers completely instead of replacing them with different invented numbers.",
                "Limited generic examples of checking logic are allowed only when they do not reveal answers to the current generated variants.",
            ],
            "validation_rules": [
                "Do not require a full key bank inside mr_intermediate HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment.",
                "Reject if mr_intermediate HTML duplicates full keys, reference answers, code solutions, stdin/stdout tests, or autocheck configs from intermediate_assessment.",
                "Judge duplication only from CHECKED MATERIAL HTML; dependency intermediate content is evidence/source context, not published mr_intermediate content.",
                "Reject if publishable mr_intermediate HTML exposes literal internal field names such as intermediate_assessment or generation_artifacts.",
                "Reject if mr_intermediate invents numeric scoring, maximum points, pass/fail threshold, percent boundary, or grade conversion not explicitly present in JSON or Markdown references.",
                "Reject if mr_intermediate mentions only '4 variants' but does not state the required per-variant composition: 5 test questions, 5 open-code questions, 5 practical code tasks, 15 total items, and at least 10 code-writing items.",
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

    if spec.kind == "current_control":
        lesson = task.get("lesson") or {}
        return {
            "contract_type": "current_control_autocheck_contract",
            "hours_raw": (lesson.get("hours") or {}).get("raw"),
            "difficulty": lesson.get("difficulty") or {},
            "required_question_count": 3,
            "generation_rules": [
                "Build internal current-control autocheck artifacts before learner-facing HTML is rendered.",
                "Create exactly 3 current-control question configs unless the source JSON explicitly requires another count.",
                "For every question, provide id, template_code, question_type, skill_target, student_prompt, correct_answers, and autocheck_config.",
                "For open-answer questions, make the prompt unambiguous and provide expected_answer_format or normalization/checking rules in autocheck_config.",
                "Keep correct answers, answer flags, matching pairs, ordering keys, normalization rules, and internal explanations inside generation_artifacts.current_control_autocheck only.",
                "The learner-facing current_control HTML must render clear questions/options/answer format but must not display correct answers, answer flags, or internal platform config.",
                "Use template_descriptions to choose platform-compatible question/template codes such as question, 6A, 6D, 6G, 8D, 10D, or 3H when appropriate.",
            ],
            "validation_rules": [
                "Do not require visible keys in learner-facing current_control HTML when generation_artifacts.current_control_autocheck contains the internal keys/config.",
                "Reject if internal current_control_autocheck artifacts are missing, have no correct answers/config, or contradict the learner-facing questions.",
                "Reject if the learner-facing HTML displays correct answers or internal autocheck configuration.",
            ],
        }

    if spec.kind == "intermediate":
        module = task.get("module") or {}
        return {
            "contract_type": "intermediate_assessment_contract",
            "required_variant_count": 4,
            "required_test_questions_per_variant": 5,
            "required_open_code_questions_per_variant": 5,
            "required_code_tasks_per_variant": 5,
            "minimum_code_item_ratio": "10/15",
            "module_lessons": module.get("lessons") or [],
            "generation_rules": [
                "Build internal assessment artifacts before learner-facing HTML is rendered.",
                "Use the module JSON and Markdown references as the source of module topics and skill coverage.",
                "For intermediate assessment, the assessment composition is governed by v34: 4 variants, each with 5 test questions, 5 open-code questions, and 5 practical code tasks.",
                "If the attestation lesson JSON gives a smaller summary count, treat it as source context for topics and timing, not as permission to omit the MaterialSpec assessment composition.",
                "For every test question, provide template_code, student_prompt, correct_answers, and autocheck_config.",
                "For each variant, among its 5 test_questions, include at least 3 distinct coded template_code values from 6A, 6D, 6G, 8D, 10D. Do not satisfy this only across the whole artifact.",
                "For every open-code question, the learner must write executable code with a verifiable result; provide student_prompt, input/output requirements where applicable, hidden_solution, rubric, and runtime_tests or manual_check_rules.",
                "For every practical code task, provide student_condition, input/output requirements, hidden_solution, and runtime_tests or manual_check_rules.",
                "Do not count output-prediction, matching, fill-gap, underlining, multiple-choice, or non-executable code-reading prompts as open-code questions.",
                "Keep correct answers, rubrics, hidden solutions, tests, and autocheck config inside generation_artifacts.intermediate_assessment only.",
                "The learner-facing intermediate HTML must render only tasks/questions/conditions and must not display keys, эталоны, hidden solutions, or internal platform config.",
                "For 6A ordering questions, options/autocheck_config.display_items are the learner-facing shuffled order. correct_answers/ordered_items are the internal correct order and must not be rendered as the visible list.",
                "For matching/pairing test questions such as 6G/8D, artifact autocheck_config.right_items is the learner-facing display order, not the answer-key order. Make it a derangement relative to left_items/correct_pairs: for every index i, right_items[i] must not be the correct pair for left_items[i].",
                "For matching/classification test questions, render the real left_items and real right_items from the artifact as two separate lists in artifact order. Do not invent generic placeholders such as Action 1, Variant A, Example 1, or similar labels unless those exact labels are the artifact items.",
                "For matching/classification test questions, learner-facing HTML may show the left and right items, but must not show solved pairs such as left — right, left -> right, left: right, or any same-position A/B item that reveals a correct pair.",
            ],
            "validation_rules": [
                "Do not require visible keys in learner-facing intermediate HTML when generation_artifacts.intermediate_assessment contains the internal keys/config.",
                "Reject if intermediate_assessment artifacts are missing, do not follow 4 x (5 test + 5 open-code + 5 practical code), lack keys/tests/rubrics, or contradict the learner-facing assessment.",
                "Reject if any variant has fewer than 3 distinct coded template_code values from 6A, 6D, 6G, 8D, 10D among its 5 test_questions.",
                "Reject if learner-facing matching/classification HTML replaces real right_items from the artifact with generic placeholders.",
                "Reject if a learner-facing 6A ordering list is identical to the internal correct order.",
                "Reject if any learner-facing matching/pairing row has list B/right_items[i] as the correct pair for list A/left_items[i]; even one same-position correct pair is answer-key leakage.",
                "Reject if learner-facing HTML displays keys, эталоны, reference answers, hidden solutions, or internal autocheck configuration.",
                "Reject if a matching/classification test question displays ready left-right pairs instead of separate item lists; that is a visible answer leak.",
            ],
        }

    if spec.kind != "practice":
        return {}

    lesson = task.get("lesson") or {}
    authoritative_task_ids = [item["id"] for item in normalized_tasks]
    return {
        "contract_type": "practice_source_contract",
        "required_task_count": len(normalized_tasks),
        "authoritative_task_ids": authoritative_task_ids,
        "hours_practice": (lesson.get("hours") or {}).get("practice"),
        "difficulty": lesson.get("difficulty") or {},
        "tasks": normalized_tasks,
        "generation_rules": [
            "Generate exactly one student-facing task variant for each task in tasks, in the same order and with the same P id.",
            "The tasks list and authoritative_task_ids are the authoritative task count, ids, order, and levels for this practice material. lesson.difficulty.*.count is planning context only and must not be used to invent extra tasks such as P6 when no corresponding lesson.practice_tasks item exists.",
            "Treat source_text as an authoritative task pattern: preserve id, level, task_type, skill_target, and constraints, but do not treat source_text as mandatory final learner wording unless it explicitly says exact wording/value is required.",
            "Create a new concrete variant of the same pattern: use different scenario, variable names, literals, input data, expected output, and code shape from theory, Markdown references, and dependency materials unless the source explicitly requires exact values.",
            "Treat subject entities inside source_text, such as 'favorite color' and 'favorite animal', as slot examples unless the source explicitly requires exact entities. Replacing them with parallel subject entities is allowed when the checked skill, number/type of variables, operation, and output structure are preserved.",
            "Do not replace the task pattern with a different skill or task type.",
            "Expected output is not invented when it is deterministically derived from explicit literals, assignments, and exact print(...) calls in source_text using standard Python print semantics.",
            "For fix/debug tasks where the corrected program behavior is deterministic from the faulty code and the intended fix, generate a concrete corrected behavior target and runtime tests for that corrected behavior. Do not choose manual_only merely because the starting code is intentionally faulty.",
            "If the source task is about reading, demonstrating, or interpreting a Python error message, the expected result/check must be the error message or diagnostic outcome, not invented normal stdout values. Use expected_error/manual checks unless the source explicitly requires corrected-program stdout.",
            "If the source or generated variant contains intentionally faulty Python code that should raise SyntaxError or NameError before correction, the learner-facing condition must name the relevant error class and either quote the expected diagnostic message or explicitly tell the learner to run the code and read that message in the IDE.",
            "For each task, visibly include learner-facing structured fields required by the practice references: level, condition, input data, output requirement, how to check, and tests/status. Do not show source_text, source task, source contract, or JSON/pipeline wording in learner-facing practice.",
            "Tests must be at least 3 explicit input -> expected output pairs when the source task defines enough information for deterministic stdout.",
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
- Learner-facing lesson materials are theory, practice, self_work, current_control, and intermediate. They must not expose lesson practice solution keys, corrected practice code, QA ids, source hashes, rc/stderr/stdout logs, internal verification traces, internal Markdown paths, source locators, local filenames, or working-folder references such as docs/..., .md, references, working area, or source-material paths.
- Autocheck keys for learner-facing self_work and current_control must not be visible in HTML. If the platform needs keys, treat them as non-rendered/internal platform configuration, not as displayed text. Do not show learner-facing key blocks, correct-answer labels, {{%answer%}}, or filled input templates with expected answers.
- For self_work, internal platform keys and autocheck settings should be carried in generation_artifacts.self_work_autocheck. They are evidence for validation and QA, not learner-facing content.
- For current_control, internal platform keys and autocheck settings should be carried in generation_artifacts.current_control_autocheck. They are evidence for validation and QA, not learner-facing content.
- For intermediate, assessment keys, reference answers, rubrics, code solutions, tests, and autocheck settings should be carried in generation_artifacts.intermediate_assessment. They are evidence for validation and QA, not learner-facing content.
- Teacher-facing artifacts are mr_theory, mr_practice, and mr_intermediate. Teacher guidance may include answers, solution keys, code snippets, rubrics, and checking guidance only when the material-specific policy permits them.
- mr_theory is teacher-facing and must be judged as teacher guidance for the theory lesson, not as a duplicate student theory material.
- mr_practice is teacher-facing and is expected to include keys/solutions for all practice tasks. Do not reject mr_practice merely because it includes teacher keys. Reject mr_practice if required teacher keys are absent, contradict the source/practice material, or leak QA/internal ids, source hashes, or service locators.
- mr_intermediate is teacher-facing publishable methodical guidance. It may explain how to conduct and check the assessment, but it must not duplicate the full variant-by-variant key bank, reference answers, hidden code solutions, stdin/stdout tests, or autocheck configs in HTML when generation_artifacts.intermediate_assessment is available.
- specification_qa is QA/internal-facing. It may include QA ids, traceability labels, tests, faulty/fixed code, and validation data. Do not reject specification_qa merely because visible QA HTML contains QA-ID labels.
- Even in specification_qa, do not render raw local source paths, tmp paths, source hashes/SHA values, or process/retry log phrasing in the visible HTML unless the user explicitly asks for a technical execution log. Use human-readable source names instead of filesystem coordinates.
- A "do not show keys" instruction applies to learner-facing materials unless the checked material kind explicitly says otherwise. The checked material kind is {spec.kind}.
""".strip()


def validation_policy_for_spec(spec: MaterialSpec) -> str:
    if spec.kind == "theory":
        return """
THEORY VALIDATION POLICY:
- Theory may include small illustrative code snippets and outputs when they explain concepts.
- Theory must not include learner self-check questions, prediction tasks, discussion tasks, or answer-key blocks.
- Theory examples must be fully explained demonstrations, not exercises for the learner to solve.
- Theory must prepare the learner for lesson.practice_tasks by explaining required concepts and typical errors, without revealing practice solutions.
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
- The required approved assessment composition must be explicit in the guidance: 4 variants; each variant has 15 items: 5 test questions, 5 open-code questions, and 5 practical code tasks; at least 10 of 15 items require writing code.
- Do not require a full key bank, full reference-answer bank, full code solutions, stdin/stdout tests, or autocheck configs inside mr_intermediate HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment.
- Reject mr_intermediate HTML if it duplicates full variant-by-variant keys, reference answers, hidden code solutions, stdin/stdout tests, or autocheck configs from the intermediate_assessment artifact.
- Judge duplication only by CHECKED MATERIAL HTML. Dependency intermediate HTML/artifacts are source evidence and must not be treated as text printed in mr_intermediate.
- Reject publishable mr_intermediate HTML if it exposes literal internal field names such as intermediate_assessment, generation_artifacts, hidden_solution, or autocheck_config. Neutral wording such as "closed teacher checking layer" is allowed.
- Reject invented scoring norms: maximum points, pass/fail thresholds, percentage boundaries, or conversion to a 5-point grade are allowed only when those exact numbers are explicitly present in JSON or Markdown references.
- When sources do not define a numeric scale, approve neutral checking workflow wording: checking by criteria groups, using the closed teacher checking layer, and fixing final results according to local organization rules without naming thresholds.
- Approve methodical guidance that explains how to conduct the assessment, how to use the internal QA/artifact layer, how to check closed/open/code tasks at a general level, how to handle appeals/typical errors, and how to keep answer keys out of student access.
- Do not reject merely because the document says keys/tests are available in the internal QA/artifact layer; that is the expected boundary for intermediate assessment.
- Reject if the document is only a pointer to the artifact and lacks useful teacher guidance for preparation, timing, checking workflow, scoring policy, and typical teacher responses.
- Reject if the document says only "4 variants" but omits the per-variant composition: 5 test questions, 5 open-code questions, 5 practical code tasks, 15 total items, and at least 10 code-writing items.
""".strip()

    if spec.kind == "specification_qa":
        return """
SPECIFICATION_QA VALIDATION POLICY:
- specification_qa is QA/internal-facing. It may contain QA ids, traceability labels, keys, tests, faulty/fixed code, and validation data.
- QA-ID labels are allowed in specification_qa visible HTML because this document is an internal QA artifact. Do not treat visible QA-ID labels as source-marker leakage for this material kind.
- Visible specification_qa HTML must not contain raw local source paths, tmp paths, source hashes/SHA values, local filenames, working-folder references such as docs/..., or Markdown source locators. Use human-readable source names if source traceability is needed.
- Do not include process/retry history as publishable QA conclusions. Phrases such as "исправлено по замечаниям валидатора", "после попытки", "validator feedback was addressed", or similar generation-loop logs are blocking unless the user explicitly requested a technical execution log.
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
- Treat STRUCTURED GENERATION ARTIFACTS.self_work_autocheck as the internal platform/QA layer. It may contain correct_answers, runtime_tests, autocheck_config, and internal_explanation.
- When self_work_autocheck contains complete internal keys/config for the visible tasks/questions, do not reject the HTML merely because the keys are not visible.
- Reject if self_work_autocheck is missing, lacks correct answers/config for generated self-check questions, or contradicts the learner-facing questions/tasks.
- Reject visible answer disclosures such as "Ключ", "Правильный вариант", explicit answer lists, {%answer%} blocks, filled {{input-text:answer}} templates, or highlighted correct options.
- Approve self-check questions without visible keys when the learner-facing task text is clear and no answers are displayed.
""".strip()

    if spec.kind == "current_control":
        return """
CURRENT_CONTROL VALIDATION POLICY:
- current_control is learner-facing. It must include visible current-control questions, but visible answer keys are forbidden in the published HTML.
- Do not require visible keys in current_control HTML. Platform/autocheck keys belong to a non-rendered/internal platform layer or QA/internal artifact, not to learner-facing text.
- Treat STRUCTURED GENERATION ARTIFACTS.current_control_autocheck as the internal platform/QA layer. It may contain correct_answers, autocheck_config, expected_answer_format, and internal_explanation.
- When current_control_autocheck contains complete internal keys/config for the visible questions, do not reject the HTML merely because the keys are not visible.
- Validate the structure against SOURCE CONTRACT FROM JSON: exactly 3 questions unless the source contract explicitly says otherwise.
- Reject if current_control_autocheck is missing, lacks correct answers/config for generated questions, has ambiguous open-answer criteria, or contradicts the learner-facing questions.
- Reject visible answer disclosures such as key blocks, correct-answer labels, explicit answer lists, {%answer%} blocks, filled {{input-text:answer}} templates, or highlighted correct options.
- Approve current-control questions without visible keys when the learner-facing task text is clear and the internal artifact contains the keys/config.
""".strip()

    if spec.kind == "intermediate":
        return """
INTERMEDIATE VALIDATION POLICY:
- intermediate is learner-facing assessment HTML. It must include the visible assessment tasks, but visible answer keys, reference answers, any rubrics/checking criteria/manual check rules, hidden solutions, runtime tests, and autocheck configs are forbidden in the published HTML.
- Treat STRUCTURED GENERATION ARTIFACTS.intermediate_assessment as the internal assessment/QA layer. It may contain correct_answers, reference_answer, rubric, runtime_tests, hidden_solution, teacher_explanation, autocheck_config, and internal_explanation.
- When intermediate_assessment contains complete internal keys/config for the visible variants, do not reject the HTML merely because keys/эталоны/tests are not visible.
- Validate the structure against SOURCE CONTRACT FROM JSON: 4 variants, each with 5 test questions, 5 open-code questions, and 5 practical code tasks.
- Open-code questions count as code only when the learner writes executable code with a verifiable result. Do not count output-prediction, matching, fill-gap, underlining, multiple-choice, or non-executable repair labels as open-code questions.
- Each variant must have at least 10 code-writing items out of 15 total items: 5 open-code questions plus 5 practical code tasks.
- Reject if intermediate_assessment is missing, lacks internal keys/rubrics/tests for generated tasks, or contradicts the learner-facing assessment.
- Reject visible answer/checking disclosures such as "Ключ", "Ключи", "Эталон", "Эталоны", "Правильный ответ", "Критерии оценивания", "Рубрика", "Правила проверки", "проверяется по коду", explicit answer lists, hidden_solution, reference_answer, correct_answers, manual_check_rules, runtime_tests, or autocheck_config inside the checked HTML.
- Candidate answer options in closed questions are allowed in learner-facing HTML. They are not answer-key leakage unless the correct option is explicitly marked or a key/answer section is printed.
- For 6A ordering questions, the visible list must use the shuffled options/display_items order. Rendering the internal correct order is answer-key leakage.
- Matching/classification questions may show the candidate left-side and right-side items, but must not show ready pairs such as "left — right", "left -> right", "left: right", or any equivalent pair map from correct_answers/autocheck_config.
- For matching/pairing questions such as 6G/8D, list B/right_items must be a derangement relative to list A/left_items and the internal correct_pairs: for every index i, right_items[i] must not be the correct pair for left_items[i]. Even one same-position correct pair is answer-key leakage.
- A 10D/6A/6D/6G/8D label in learner-facing HTML is a visible task-type label only. Do not require platform-import markup such as {{input-text:...}} in publishable HTML when intermediate_assessment carries the structured template_code and autocheck_config.
- "Find/fix/explain the error" assessment tasks are allowed when the learner-facing HTML shows only the faulty code or condition. Treat them as leakage only when the corrected/reference solution is visible in HTML.
- Do not treat the attestation lesson JSON summary count as overriding skill 05 composition. Use it for topic/timing context unless the source contract says otherwise.
""".strip()

    if spec.kind != "practice":
        return "No additional material-kind validation policy."

    return """
PRACTICE VALIDATION POLICY:
- Practice generation is template-based. lesson.practice_tasks defines the authoritative pattern: P id, level, task type, target skill, and constraints. It is not necessarily the final learner wording.
- SOURCE CONTRACT FROM JSON.authoritative_task_ids is the authoritative task set for learner-facing practice. Do not require tasks outside that list because of lesson.difficulty.*.count, module totals, reference examples, or generic L1/L2 proportions. A mismatch between difficulty counts and lesson.practice_tasks is a source-data warning, not permission to invent P6/P7.
- Approve a new task variant when it preserves the source pattern but uses different scenario, variable names, literals, input/output data, and code shape from theory, Markdown references, and dependency materials.
- Reject direct copying from theory/references/dependencies when the copied content becomes the learner-facing practice task, starter code, test data, or expected output without an explicit source requirement.
- Do not require the practice task to preserve source_text verbatim. Require pattern faithfulness, not textual identity.
- Treat source_text subject entities as replaceable slot examples unless the JSON/prompt/reference explicitly requires exact entities. For example, "favorite color" and "favorite animal" may become other parallel categories when the task still checks two string variables and printing both values.
- Validate practice methodology, not a rigid task-layout template: check that tasks reveal the lesson topic, build the intended skill progression, are understandable for learners, are internally consistent, and have a reasonable checking path.
- Do not reject practice solely because a task lacks a specific visual subsection such as "Код", "Код в редакторе", starter code, or a standalone <pre><code> block. For write-code tasks, a clear condition plus tests/checking path is sufficient; starter code is optional unless the source task type is fix/debug or explicitly requires given code.
- Practice references require at least 3 visible input -> expected output pairs when the source task provides enough information for deterministic stdout.
- Visible expected stdout in a student-facing deterministic test table is allowed and expected for practice materials. It is a test oracle, not a forbidden answer key, as long as it does not include corrected code, hidden_solution text, teacher_explanation, QA ids, source hashes, or internal trace data.
- Do not reject practice merely because deterministic tests show concrete expected stdout. Reject only when the material reveals the corrected code, tells the exact edit operation, exposes hidden_solution/teacher_explanation, or invents unsupported expected output.
- If the source task is to read, demonstrate, or interpret a Python error message, reject stdout-only tests with normal output values. The check/result must correspond to the task: expected_error/error message/diagnostic outcome, or a manual/static check when the platform cannot autocheck stderr/traceback.
- For fix/debug tasks, do not validate learner-facing faulty_code, faulty_code_display, or starter_code for code correctness, syntactic validity, runtime success, or whether it already produces the expected output. These fields are intentionally faulty learning inputs.
- Do not reject a fix/debug practice merely because the faulty code contains an unclosed string/bracket/quote, typo, NameError/SyntaxError cause, or another defect the learner is supposed to find. Reject only if the debugging objective is incoherent, the expected behavior/tests contradict the task, or the exact fix/corrected code is visible to the learner.
- For tasks about SyntaxError / unterminated string literal / EOL while scanning string literal, do not reject merely because the displayed faulty code visually spans the next line, makes a following line look like part of the open string, or would not parse as a clean isolated one-error snippet. That is an expected consequence of the target error. Judge whether the task is understandable, copyable, and does not reveal the exact fix.
- For refactoring tasks, separate runtime tests from manual/static checks. Runtime tests verify behavior preservation through stdin -> expected stdout. Manual/static checks verify requirements that stdout cannot prove: better variable names, comments, removed duplicated computation, named constants, and code readability.
- Do not reject a refactoring practice merely because stdout tests cannot prove variable names, comments, or duplication removal when the learner-facing material includes explicit manual/static checks for those requirements.
- Reject a refactoring practice when it has neither runtime tests for behavior that is actually runnable nor manual/static checks for non-runtime requirements.
- Do not require source_text, source task, source contract, JSON wording, or pipeline trace to be visible in learner-facing HTML. Those are internal artifacts for validation and MR/QA. Their absence from student practice is correct.
- Do not treat an expected output as invented when it follows deterministically from explicit source_text literals, assignments, and exact print(...) calls using standard Python print semantics.
- For source tasks that are underspecified for deterministic stdout, do not require fake values or fake expected output. The material may mark tests as absent/not applicable until source clarification, but it must preserve the same visible learner-facing task structure: level, condition, input data, output requirement, how to check, and tests/status.
- For underspecified source tasks, an honest "tests are absent/not applicable until source clarification" status is valid and should be approved when the task meaning is preserved and no fake expected output, fake variable names, or fake values are introduced.
- Do not reject merely because the material calls an underspecified task "training", "manual check", "without autocheck", or "until source clarification". Treat that wording as non-blocking editorial risk unless it changes the task, hides required fields, or invents a fake test.
- If the source text lists two entities, for example "favorite color" and "favorite animal", restating that as "two variables" is allowed, and do not require those exact entity nouns unless exact entities are explicitly source-required. A variant may use other parallel entities when it preserves the checked skill, number/type of variables, output operation, and absence/presence of input.
- For an underspecified source task, do not require the tests/status block to use the same input -> expected output table as deterministic tasks. A fake test row, placeholder expected output, dash, or "not specified" value inside an expected-output column is invalid.
- Do not create an impossible requirement where an underspecified task must both avoid invented values and still provide exact autocheck pairs.
- Separate deterministic tasks from underspecified tasks in the report. A problem in one task does not make valid tests in another task blocking.
""".strip()


def material_result_summary(
    material: MaterialResult,
    *,
    max_content_chars: int = 12000,
    include_content: bool = True,
    include_runtime_metadata: bool = False,
) -> dict[str, Any]:
    summary = {
        "kind": material.kind,
        "type": material.material_type,
        "status": material.status,
    }
    if material.generation_artifacts:
        summary["generation_artifacts"] = material.generation_artifacts
    if include_runtime_metadata:
        summary["agent"] = material.agent_type
        summary["iterations"] = material.iterations
        summary["validation_issues"] = list(material.validation_issues)
        summary["agent_notes"] = list(material.agent_notes)
    if include_content:
        summary["content"] = material.content[:max_content_chars]
    else:
        summary["content_chars"] = len(material.content)
        summary["content_omitted"] = True
    return summary


def dependency_result_summaries_for_validation(
    spec: MaterialSpec,
    dependencies: list[MaterialResult],
) -> list[dict[str, Any]]:
    if spec.kind == "mr_intermediate":
        summaries = [
            material_result_summary(item, include_content=False)
            for item in dependencies
        ]
        for summary in summaries:
            if summary.get("kind") == "intermediate":
                summary["content_omission_reason"] = (
                    "Dependency intermediate HTML is not the checked mr_intermediate material. "
                    "Use dependency generation_artifacts for internal keys/config only; judge duplication "
                    "only against CHECKED MATERIAL HTML."
                )
        return summaries
    return [material_result_summary(item) for item in dependencies]


def package_material_payload(material: MaterialResult) -> dict[str, Any]:
    content = material.content
    stripped = content.strip()
    lower = stripped.lower()
    generation_artifacts = material.generation_artifacts
    structured_keys = tuple(
        key
        for key in PRIMARY_STRUCTURED_ARTIFACT_KEYS_BY_KIND.get(material.kind, ())
        if key in generation_artifacts
    )
    payload = {
        "kind": material.kind,
        "type": material.material_type,
        "status": material.status,
        "content_chars": len(content),
        "html_diagnostics": {
            "starts_with_style": stripped.startswith("<style>"),
            "contains_cc_lesson": '<div class="cc-lesson">' in content,
            "ends_with_final_div": stripped.endswith("</div>"),
            "style_tag_count": lower.count("<style>"),
            "open_div_count": lower.count("<div"),
            "close_div_count": lower.count("</div>"),
        },
        "generation_artifacts": generation_artifacts,
        "primary_structured_artifact_keys": structured_keys,
    }
    if structured_keys:
        payload["full_final_content_omitted"] = True
        payload["content_omission_reason"] = (
            "This material has primary structured artifacts. Package LLM validation must not semantically "
            "review rendered HTML; HTML can be regenerated from artifacts by a technical renderer."
        )
    else:
        payload["content_truncated"] = False
        payload["full_final_content"] = content
    return payload


def validation_result_summary(validation: ValidationResult) -> dict[str, Any]:
    return {
        "approved": validation.approved,
        "issues": list(validation.issues),
        "fix_instructions": list(validation.fix_instructions),
        "issues_by_block": list(validation.issues_by_block),
        "passed_blocks": list(validation.passed_blocks),
    }


def generation_artifacts_for_validation(
    spec: MaterialSpec,
    generation_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    if not generation_artifacts:
        return {}
    return dict(generation_artifacts)


PRIMARY_STRUCTURED_ARTIFACT_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "practice": ("practice_templates", "practice_instances"),
    "self_work": ("self_work_autocheck",),
    "current_control": ("current_control_autocheck",),
    "intermediate": ("intermediate_assessment",),
}


def primary_structured_artifact_keys(spec: MaterialSpec, artifacts: dict[str, Any]) -> tuple[str, ...]:
    keys = PRIMARY_STRUCTURED_ARTIFACT_KEYS_BY_KIND.get(spec.kind, ())
    return tuple(key for key in keys if key in artifacts)


def has_primary_structured_artifacts(spec: MaterialSpec, artifacts: dict[str, Any]) -> bool:
    return bool(primary_structured_artifact_keys(spec, artifacts))


def validation_target_mode(spec: MaterialSpec, artifacts: dict[str, Any]) -> str:
    return "structured_artifacts" if has_primary_structured_artifacts(spec, artifacts) else "html"


def validation_html_section(spec: MaterialSpec, content: str, artifacts: dict[str, Any]) -> str:
    if has_primary_structured_artifacts(spec, artifacts):
        return (
            "RENDERED HTML IS NOT INCLUDED IN THIS LLM VALIDATION PROMPT.\n"
            "For this material kind, LLM semantic validation must use STRUCTURED GENERATION ARTIFACTS as the "
            "checked material. HTML is a renderer output and is outside LLM semantic validation/controller scope."
        )
    return f"""CHECKED MATERIAL HTML START
{content}
CHECKED MATERIAL HTML END"""


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
    previous_validation: ValidationResult | None = None,
) -> str:
    previous_validation_payload = (
        validation_result_summary(previous_validation)
        if previous_validation is not None
        else {}
    )
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

PREVIOUS FULL VALIDATION RESULT:
{compact_json(previous_validation_payload)}

VARIANT RULES:
- Return PracticeTaskInstanceSet structured output only.
- Always fill PracticeTaskInstanceSet.lesson_goal and PracticeTaskInstanceSet.lesson_objectives. They are rendered as the student-facing "Цели и задачи" block. Use operational learner actions aligned with the topic, for example: runs code in the cloud IDE, reads SyntaxError/NameError diagnostics, fixes one error, checks the result.
- Create exactly one PracticeTaskInstance for every PracticeTaskTemplate, in the same order and with matching id/template_id.
- Preserve the template's level, task_type, skill_target, and invariants.
- Generate a new concrete scenario and new values: variable names, literals, input data, expected output, and code shape must differ from theory, Markdown references, and dependency materials unless the source explicitly requires exact values.
- Subject entities in source_text are slot examples unless explicitly marked exact. You may replace them with parallel entities when the checked skill, number/type of variables, output operation, and constraints remain the same.
- Do not change the task into another skill or task type.
- Treat source_text/source task as internal pattern evidence. Never copy it into scenario, student_condition, input_requirements, or output_requirements, and never label learner-facing text as "source task", "from JSON", or "as in the lesson task".
- For retry, if previous issues mention source_text, source task, source contract, exact-fix hints, or student-facing fields revealing the fix, rewrite only the affected PracticeTaskInstance fields and keep valid task ids/order/tests where possible.
- For fix/debug tasks, the learner-facing fields scenario, student_condition, input_requirements, and output_requirements must not reveal the exact edit operation. Do not say "add a quote", "replace X with Y", "the quote is missing", "the function name is misspelled", or similar solution hints. The learner may see the faulty code, the error message when source-required, the expected behavior, and the test result target; the exact fix belongs only in hidden_solution and teacher_explanation.
- For fix/debug tasks with intentionally faulty code, put the raw code into faulty_code. If a separate learner-facing rendering is needed, put only the code the learner should copy into faulty_code_display.
- faulty_code_display and starter_code must contain only code lines that the learner should copy into the editor. Do not add marker/comment lines, neutral hints, or explanatory comments such as "# fragment intentionally breaks here" or "# фрагмент намеренно обрывается здесь" inside the code block.
- Leave display_note empty unless a source-required, neutral formatting note is unavoidable. Never use display_note to point at where the error is, explain that a fragment is intentionally incomplete, or give a meta-comment about how the faulty code is broken.
- Use starter_code for ordinary starter code or as a backward-compatible learner-facing code field. When faulty_code_display is present, PracticeMaterialAgent must render faulty_code_display instead of raw faulty_code.
- Put solutions, corrected code, answer keys, and teacher-only explanations only into hidden_solution and teacher_explanation. These fields are internal artifacts for MR/QA and must not be shown in learner HTML.
- Classify each fix/debug task before choosing run_mode:
  1. diagnostic task: the source explicitly asks to read, demonstrate, identify, or interpret a Python error message. Use manual_only or expected_error/error_message checks; student_condition must name the relevant error class and either quote the expected diagnostic message or instruct the learner to run the code and read it in the IDE.
  2. correction task with deterministic corrected behavior: the source gives faulty code and a clear intended correction/output can be derived from literals, variables, print(...) calls, or the variant you created. Use run_mode=single_file, provide hidden_solution, output_requirements with exact corrected stdout, and create runtime_tests/tests for that corrected behavior. Do not downgrade this to manual_only just because the initial code is faulty.
  3. truly underspecified task: only if neither the diagnostic outcome nor corrected behavior can be made source-faithfully checkable, use manual_only or needs_platform_clarification and explain the manual/clarification policy.
- If the source pattern asks the learner to read, demonstrate, or interpret a Python error message such as SyntaxError or NameError, the result/check must be the expected error message or diagnostic outcome. Do not invent normal stdout values for such a task. Use expected_error/error_message in runtime_tests/tests when the platform can check the error text, or keep runtime_tests/tests empty and put the diagnostic check into manual_checks/output_requirements.
- If deterministic runtime checks are source-supported for normal program behavior, create at least 3 meaningful stdin -> expected stdout checks, put them into runtime_tests, and mirror the same list in legacy tests. Include typical and boundary/special cases when applicable. Every stdout item must use exactly the keys input and expected_output. Preserve trailing newlines in expected_output as "\\n"; do not use keys such as output/stdout/result.
- For no-stdin corrected-output fix/debug tasks, create 3 runtime_tests/tests with empty input and the same exact expected_output. This repetition is acceptable because the platform still needs explicit test rows and the program has deterministic output without input.
- For refactoring tasks, use runtime_tests only to verify behavior preservation. Put requirements that stdout cannot prove into manual_checks: better variable names, comments, removed duplicated computation, named constants, and readability.
- Set run_mode to single_file when one runnable file is intended, separate_snippets when subtasks should be run independently, manual_only when only manual/static checking is meaningful, or needs_platform_clarification when the source does not define how the platform runs multi-part code.
- If deterministic runtime tests are not source-supported, keep runtime_tests and tests empty and state the manual/clarification policy in output_requirements, manual_checks, hidden_solution, and teacher_explanation instead of inventing fake exact outputs.
- In uniqueness_notes, explicitly state what differs from theory/references/dependencies.

RETRY REPAIR RULES:
- If PREVIOUS FULL VALIDATION RESULT.approved is false, treat this as a repair attempt. Do not return the same PracticeTaskInstanceSet unchanged.
- Use PREVIOUS FULL VALIDATION RESULT.issues, issues_by_block, fix_instructions, and passed_blocks to decide what must change.
- Preserve blocks listed in passed_blocks unless a document-level fix requires a narrow adjacent change.
- Repair every block listed in blocking issues_by_block and every document-level issue. If the issue is "Цели и задачи" or missing lesson objectives, update lesson_goal/lesson_objectives; do not try to solve that by changing only task text.
- If a previous issue says tasks are manual_only without tests but the source-supported corrected behavior is deterministic, change the affected task instances to run_mode=single_file and provide runtime_tests/tests for corrected behavior.
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


def build_current_control_autocheck_prompt(
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
Build the internal current-control checking/autocheck artifact. Do not generate learner-facing HTML.

AGENT TYPE:
CurrentControlAutocheckAgent

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

PREVIOUS CURRENT_CONTROL GENERATION ARTIFACTS:
{compact_json(previous_artifacts or {})}

PREVIOUS VALIDATION/STRUCTURAL ISSUES:
{compact_json(previous_issues)}

CURRENT-CONTROL ARTIFACT RULES:
- Return CurrentControlAutocheckSet structured output only.
- Create exactly 3 questions unless the source JSON explicitly requires another count.
- The questions field must contain only CurrentControlAutocheckQuestion objects. Never put notes, strings, markdown, or "agent_notes" entries inside questions.
- Put diagnostic notes only into the top-level agent_notes field as a list of strings.
- Each question must include id, template_code, question_type, skill_target, student_prompt, correct_answers, and autocheck_config.
- Use template_descriptions to choose platform-compatible template_code values. Prefer a varied mix when the references require multiple template kinds.
- For closed questions, provide options and an autocheck_config that identifies the correct option(s).
- For open-answer questions, make the student_prompt unambiguous and provide expected_answer_format plus autocheck_config normalization or matching rules. If "print" and "print()" could both be considered defensible, rewrite the prompt or list all acceptable normalized answers.
- Put correct answers, answer flags, matching pairs, ordering keys, normalization rules, and teacher-only explanations only in this internal artifact.
- Do not instruct the HTML generator to display correct_answers, answer flags, filled input templates, autocheck_config, or internal_explanation.
- On retry, preserve valid previous questions where possible and repair only items connected to previous issues.
""".strip()


def build_current_control_autocheck_system_prompt() -> str:
    return (
        "You are CurrentControlAutocheckAgent. Build internal checking and autocheck configuration for "
        "current_control. Do not produce learner-facing HTML. Return only the configured structured output."
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
- For each variant, create exactly 5 test_questions, 5 open_code_questions, and 5 code_tasks.
- For every test question, include id, template_code, skill_target, student_prompt, correct_answers, and autocheck_config.
- A test question must have unambiguous grading. If more than one answer is defensible, either rewrite the prompt/options so exactly one answer is correct, or use a template/autocheck_config that supports multiple correct answers and list every correct answer. Never choose one arbitrary answer while internal_explanation says several answers are valid.
- For each variant, among the 5 test_questions, include at least 3 distinct coded template_code values from this set: 6A, 6D, 6G, 8D, 10D. This is a per-variant requirement, not a whole-artifact average.
- Other closed question types such as single_choice or multiple_choice are allowed, but they do not count toward the 3 required coded template types.
- For every open-code question, include id, skill_target, student_prompt, input_requirements, output_requirements, hidden_solution, rubric, and runtime_tests or manual_check_rules. The student_prompt must require the learner to write executable code with a verifiable result.
- For every practical code task, include id, skill_target, student_condition, input_requirements, output_requirements, hidden_solution, and runtime_tests or manual_check_rules.
- When runtime_tests are used for open-code questions or practical code tasks, include at least 3 tests with typical and boundary/special cases where applicable.
- Runtime tests must describe the corrected/reference behavior, not the faulty starter_code behavior. If output_requirements says exactly two decimal places / two digits after the decimal point, every expected stdout value must contain exactly two decimal digits, for example 85.00 rather than 85.0.
- If output_requirements requires exact decimal-place formatting, hidden_solution must use formatting that preserves trailing zeroes, such as :.2f or format(..., ".2f"). round(..., 2) alone is not enough for exact two visible decimal places.
- Do not put hints in learner-facing fields. Do not turn open-code questions into output-prediction, matching, fill-gap, underlining, or multiple-choice tasks.
- Use module.lessons and Markdown references for topic coverage. Do not copy prior examples verbatim when a new assessment item can be created.
- Keep correct answers, reference answers, rubrics, manual_check_rules, hidden solutions, tests, and autocheck configs only in this internal artifact.
- The HTML renderer will use student_prompt/options/student_condition/starter_code/input_requirements/output_requirements but must not display correct_answers, hidden_solution, teacher_explanation, internal_explanation, rubric, manual_check_rules, runtime_tests, or autocheck_config.
- The HTML renderer must not create learner-facing sections titled "Критерии оценивания", "Рубрика", or "Правила проверки", and must not print phrases such as "это проверяется по коду".
- For 6A ordering questions, options/autocheck_config.display_items are the visible shuffled order. correct_answers, ordered_items, items_in_correct_order, and correct_order are internal keys and must not be used as the visible item order.
- For matching/pairing test questions such as 6G/8D, autocheck_config.left_items and right_items are display lists. Put right_items in deranged display order relative to left_items/correct_pairs: for every index i, right_items[i] must not be the correct pair for left_items[i]. correct_pairs remains the internal key.
- For matching/classification test questions, the HTML renderer must use the real left_items/right_items from the artifact and render them as two separate lists in artifact order. Do not replace them with placeholders such as Action 1, Variant A, Example 1, or similar generic labels.
- On retry, preserve valid previous variants/items where possible and repair only items connected to previous issues.
- If previous issues say list B/right_items is in the same order as list A/left_items, leaks matching pairs by position, or has same-position correct pairs, repair only matching/pairing questions by reordering right_items display order into a derangement; keep left_items and correct_pairs semantically unchanged.
- If previous issues say a 6A ordering list is already in the correct order, repair only that 6A question by reordering options/display_items; keep correct_answers/ordered_items semantically unchanged.
- If previous issues say "must include at least 3 coded template types", repair only test_questions in the named variant(s). Keep exactly 5 test_questions and replace enough single_choice/multiple_choice template_code values with suitable distinct codes from 6A, 6D, 6G, 8D, 10D; keep correct_answers/autocheck_config consistent with the selected template_code.
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
    validation_generation_artifacts = generation_artifacts_for_validation(spec, generation_artifacts)
    target_mode = validation_target_mode(spec, validation_generation_artifacts)
    primary_keys = primary_structured_artifact_keys(spec, validation_generation_artifacts)
    html_section = validation_html_section(spec, content, validation_generation_artifacts)
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
      "field_path": "practice_instances.tasks[P3].student_condition",
      "severity": "blocking",
      "issue": "what is wrong in this block",
      "fix_instruction": "what to change in this block",
      "evidence_quote": "exact short quote copied from CHECKED MATERIAL HTML only for rendered HTML issues"
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
The deterministic rule validator checks only technical invariants: response boundary, allowed HTML container, dangerous tags, package statuses, and dependency order.
You are responsible for semantic validation. Evaluate the material against the JSON, prompt/skill files, Markdown references, material kind, dependencies, and validation issues.
Check semantic completeness, internal consistency, source faithfulness, sufficient depth for the configured academic hours, suitability for the target learner/teacher, required task/assessment composition, and absence of contradictions or invented facts.
Do not reject a material only because section headings use different wording, unless a prompt/skill/reference explicitly requires exact fixed headings. Judge whether the required meaning and content are present.

PRIMARY VALIDATION TARGET:
- If VALIDATION TARGET MODE is structured_artifacts, validate only STRUCTURED GENERATION ARTIFACTS semantically. Do not validate rendered HTML with LLM.
- For practice, the primary content is practice_templates and practice_instances. Validate task ids, task fields, tests, manual checks, hidden solutions, teacher explanations, and internal consistency from practice_instances. Report fixes by exact field_path such as practice_instances.tasks[P3].student_condition.
- For self_work, current_control, and intermediate, validate the corresponding autocheck/assessment artifact as the primary platform/QA layer when present.
- If VALIDATION TARGET MODE is html, validate CHECKED MATERIAL HTML as the primary content.
- If VALIDATION TARGET MODE is structured_artifacts, any material-kind policy text about visible HTML/rendering is out of LLM semantic validation scope. Apply only the parts that map to structured student-facing/internal fields.
- Do not ask for deterministic string surgery on HTML when the defect belongs to structured content. Point to the structured field that must be regenerated.

VALIDATION COMPLETENESS AND STABILITY:
- Perform a full audit of the primary checked material and all applicable source requirements before returning the structured decision.
- On the first validation pass, report every material issue you can identify for this checked material. Do not intentionally defer lower-priority, later-block, formatting, methodology, task-composition, source-faithfulness, or consistency issues to later retries.
- Do not stop after finding the first blocking issue. If several blocks have the same defect, report either one grouped issue naming all affected blocks or one issues_by_block item per affected block.
- Keep the validation standard stable across attempts: do not introduce a new requirement on a later attempt unless it was genuinely not applicable to the earlier content or is caused by a newly generated change.
- Top-level issues must be a complete summary of all blocking issues, not only the most important one. Non-blocking issues that would be useful for repair should still be listed as non-blocking issues_by_block entries.
- Before setting approved=true, explicitly verify that no known blocking issue remains under the material-kind policy, channel/key visibility policy, source contract, and prompt/skill requirements.

MATERIAL-KIND VALIDATION POLICY:
{validation_policy_for_spec(spec)}

{channel_key_visibility_policy_for_spec(spec)}

BLOCK-LEVEL REPORTING:
- For every blocking issue, add a corresponding object to issues_by_block.
- Use the nearest HTML heading id as block_id, for example "#selfcheck"; if there is no id, use the heading text.
- If the issue belongs to structured content, fill field_path with the exact JSON path that should be repaired. Keep block_id as the user-visible task/block id when available, for example "#P3".
- Do not put a block into passed_blocks if it has any issue that requires editing.
- Add blocks to passed_blocks when they are semantically valid and should be preserved during retry.
- Keep fix_instructions actionable and localized to the named block when possible.

EVIDENCE RULE:
- For issues about structured content, field_path is required and evidence_quote may be empty.
- For issues about rendered HTML, copied visible text, visible leaked key, visible formatting, or a specific HTML block, include an exact short evidence_quote copied from CHECKED MATERIAL HTML. This applies only when CHECKED MATERIAL HTML is included in this prompt.
- The evidence_quote must contain the claimed rendered defect itself, not only the surrounding heading.
- If rendered HTML is not included in this prompt, do not report rendered HTML issues.
- Do not use HTML evidence rules to avoid reporting structured defects. If the contradiction is in practice_instances/current_control_autocheck/intermediate_assessment, report the structured field_path directly.
- Top-level issues should either refer to field_path or to the issues_by_block item that contains evidence_quote.

AGENT TYPE:
MaterialValidatorAgent

CHECKED MATERIAL KIND:
{spec.kind}

CHECKED MATERIAL TYPE:
{spec.material_type}

VALIDATION TARGET MODE:
{target_mode}

PRIMARY STRUCTURED ARTIFACT KEYS:
{compact_json(primary_keys)}

PROMPT/SKILL FILES FOR CHECKED MATERIAL:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json(dependency_result_summaries_for_validation(spec, dependencies))}

STRUCTURED GENERATION ARTIFACTS FOR CHECKED MATERIAL:
{compact_json(validation_generation_artifacts)}

ARTIFACT VISIBILITY RULE:
- For current_control, current_control_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. If it is complete and consistent, it satisfies requirements for keys/autocheck without visible answer keys in HTML.
- The section above is structured generation output, not learner-facing HTML. It is the primary validation target for structured material kinds and may include teacher-only/internal fields.
- Do not infer a learner-facing answer leak merely because an internal artifact contains keys, hidden_solution, teacher_explanation, correct_answers, tests, or autocheck_config. For structured validation, a leak is present only when forbidden internal content appears in structured student-facing fields.
- For self_work, self_work_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. If it is complete and consistent, it satisfies requirements for keys/autocheck without visible answer keys in HTML.
- For intermediate, intermediate_assessment is the required non-rendered assessment/QA layer for keys, эталоны, rubrics, tests, and solutions. If it is complete and consistent, it satisfies requirements for keys/autocheck without visible answer keys in HTML.
- For mr_intermediate, dependency intermediate content/artifacts are source evidence only. They are not the checked material. Claims that mr_intermediate duplicates a full KIM/variant bank are valid only when the duplicated variants/tasks appear between CHECKED MATERIAL HTML START/END.

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}

{html_section}

If VALIDATION TARGET MODE is structured_artifacts, only STRUCTURED GENERATION ARTIFACTS is the LLM-checked material.
If VALIDATION TARGET MODE is html, CHECKED MATERIAL HTML is the LLM-checked material.
Do not treat the JSON response schema example above as material content.
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
    validation_generation_artifacts = generation_artifacts_for_validation(spec, generation_artifacts)
    target_mode = validation_target_mode(spec, validation_generation_artifacts)
    primary_keys = primary_structured_artifact_keys(spec, validation_generation_artifacts)
    html_section = validation_html_section(spec, content, validation_generation_artifacts)
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
- If VALIDATION TARGET MODE is structured_artifacts, audit validator claims against STRUCTURED GENERATION ARTIFACTS, not rendered HTML.
- For validator issues about structured content, require a concrete field_path or an equivalent exact reference to the structured object/field.
- If VALIDATION TARGET MODE is structured_artifacts, overrule validator issues about rendered HTML structure, visible HTML formatting, HTML section composition, or HTML leakage. HTML rendering is a separate technical layer and is not a controller problem.
- If VALIDATION TARGET MODE is html, require a direct quote from CHECKED MATERIAL HTML for rendered HTML claims.
- Treat a learner-facing leakage claim as unproven if it relies only on internal artifacts and does not show that forbidden content appears in structured student-facing fields or, for html mode only, in CHECKED MATERIAL HTML.
- Overrule a validator issue when the claim is unsupported by the checked material, based on a debatable interpretation, based on optional style, or fixable by a narrow local edit without changing the material's source meaning.
- Do not let one local wording or terminology imperfection fail the whole material unless that imperfection teaches a materially wrong core rule, breaks required task composition, reveals forbidden answers, or makes the artifact unsafe/unusable.
- Distinguish educational simplification from factual error. A simplified beginner-facing statement is acceptable when it is pedagogically useful, does not contradict the course sources, and does not lead the learner to solve tasks incorrectly.
- Distinguish recommended course convention from language/platform specification. A convention can be stated as the course's recommended practice without being treated as a false claim about all possible syntax.
- If the validator found a real but local problem, prefer approve_material with non_blocking_issues and fix_instructions when the rest of the material is usable and the fix does not require regenerating the whole artifact.

DECISION POLICY:
- Never approve if deterministic technical checks failed, if content is empty, unsafe, malformed, or outside the requested material kind.
- Keep failed if there are major source contradictions, invented facts that change the task, missing core content, wrong audience, insufficient depth for configured hours, missing required task/assessment composition, or unresolved dependency problems.
- If VALIDATION TARGET MODE is structured_artifacts, do not keep failed for HTML rendering/assembly issues. Those belong to a separate deterministic render/smoke-test step and can be fixed by regenerating HTML from the approved artifacts.
- You may approve when remaining objections are editorial preferences, harmless wording choices, overly literal interpretation, optional/recommended style, non-blocking section naming differences, unproven "potential" concerns, or minor imperfections that do not harm the artifact purpose.
- If SOURCE CONTRACT FROM JSON contains authoritative_task_ids, validator objections about missing keys/content for task ids outside that list are invalid and should be overruled. Do not let non-authoritative task ids block an otherwise usable material.
- For learner-facing practice, SOURCE CONTRACT FROM JSON.authoritative_task_ids is authoritative for task count, ids, order, and levels. If lesson.difficulty.*.count conflicts with lesson.practice_tasks, treat that as source-data inconsistency and overrule validator demands to invent extra tasks such as P6/P7 when all authoritative_task_ids are present in the material.
- You may approve when the material has a small number of local factual/terminology imprecisions that are easy to correct and do not undermine the central teaching objective; record them as non_blocking_issues with precise fix_instructions.
- Keep failed for factual errors only when they are central, repeated, source-contradicting, or likely to make the learner form a wrong mental model that affects task performance. Do not keep failed for a single localized nuance if the surrounding explanation is still usable.
- For practice materials, overrule validator objections that treat visible deterministic expected stdout as a forbidden key. Expected stdout in a test table is allowed when it is part of the learner's platform test description and does not reveal corrected code or the exact edit operation.
- For learner-facing practice materials, overrule validator objections that treat teacher-only fields inside internal generation artifacts as learner-facing leaks. A hidden solution/explanation leak is blocking only when it appears in the checked learner-facing target: structured student-facing fields for structured mode, or checked HTML for html mode.
- For learner-facing practice materials, overrule validator objections that enforce a concrete task-formatting/UI requirement such as mandatory "Код"/"Код в редакторе" sections, starter-code blocks, or standalone <pre><code> editor placeholders. The validator should judge methodological correctness and topic coverage, not require one specific rendering structure, unless the source task type explicitly requires given code.
- For practice fix/debug tasks, overrule validator objections that treat intentionally faulty starter/faulty_code/faulty_code_display as invalid because it is not syntactically correct, does not run, raises NameError/SyntaxError, has an unclosed quote/bracket/string, or does not already produce the expected output. Keep failed only when the learner-facing material reveals the exact fix/corrected code, the task objective is incoherent, or tests/expected behavior contradict the task.
- For practice tasks about SyntaxError / unterminated string literal / EOL while scanning string literal, overrule validator objections that the displayed faulty snippet visually becomes multi-line, makes the next line look consumed by the open string, or is not a clean "one isolated error" parse. This is the expected learner-facing faulty input for this error type, not a generation defect, unless the material exposes the corrected code or gives the exact edit.
- For specification_qa, overrule validator objections that visible QA-ID labels are internal marker leakage. QA-ID is allowed in this internal QA artifact. Keep failed for specification_qa process/retry logs, raw local source paths, source hashes/SHA values, contradictory keys/tests, invented task ids, or visible content that would be unsafe if copied into learner/teacher materials.
- For learner-facing current_control, overrule validator objections that require visible keys when generation_artifacts.current_control_autocheck contains complete internal correct_answers/autocheck_config consistent with the HTML. Keep failed if this artifact is missing, incomplete, contradictory, or if HTML visibly displays the keys.
- For practice tasks that are underspecified for deterministic stdout, prefer approving when the material preserves the source task, avoids fake values/expected output, and clearly marks tests as absent/not applicable/manual/unavailable until clarification. Do not keep failed merely because the material uses wording such as "training task", "manual check", "without autocheck", or "until source clarification".
- For practice wording, do not treat source subject-entity substitutions as inventions when they preserve the same checked skill and task structure. If the source lists "favorite color" and "favorite animal", a variant with other parallel categories is acceptable unless the source explicitly requires those exact entities.
- For intermediate, overrule validator objections that treat candidate answer options as answer-key leakage. Closed questions must show answer options; this is blocking only when HTML explicitly marks the correct option or prints a key/answer section.
- For intermediate, keep failed when a matching/classification question displays solved left-right pairs such as "left — right", "left -> right", or "left: right", or when any list B/right_items[i] is the correct pair for list A/left_items[i]. Candidate items are allowed, but ready pair maps or even one same-position correct pair are answer-key leakage.
- For intermediate, overrule validator objections that require platform-import template markup such as {{input-text:...}} in publishable learner HTML when generation_artifacts.intermediate_assessment contains template_code and autocheck_config. The HTML is the readable student artifact; the internal artifact is the platform/autocheck layer.
- For intermediate, overrule validator objections that treat "find/fix/explain the error" prompts as leaked solutions merely because the error is local or obvious. Keep failed only when the corrected/reference solution itself is visible in HTML.
- For mr_intermediate, overrule validator objections that require the full intermediate key bank, reference answers, code solutions, stdin/stdout tests, or autocheck configs to be duplicated in HTML when the approved intermediate dependency contains generation_artifacts.intermediate_assessment. The expected HTML is methodical guidance plus instructions for using the internal artifact, not the full answer bank.
- For mr_intermediate, keep failed when the HTML duplicates the full answer bank/tests/solutions from intermediate_assessment or lacks practical methodical guidance.
- For mr_intermediate, keep failed when the checked HTML invents maximum points, pass/fail thresholds, percentage boundaries, or conversion to a 5-point grade that are not explicitly present in JSON or Markdown references. This is a source contradiction, not an editorial issue.
- For mr_intermediate, keep failed when the checked HTML does not explicitly state the approved composition of each variant: 5 test questions, 5 open-code questions, 5 practical code tasks, 15 total items, and at least 10 code-writing items.
- For theory boundary issues, require the validator to prove copyable answer leakage, not just conceptual similarity. Conceptual overlap with practice skills is expected and allowed in theory.
- A theory example is blocking only when it is a complete solved answer to a specific practice task and materially reuses source-specific variables, literals, target phrase, exact expected output, or the full task structure from that practice task.
- Generic lesson concepts and neutral explained demonstrations are not answer leakage: print("text"), print(variable), print("label:", value), print with comma-separated arguments, and text-vs-variable comparisons may be necessary explanations.
- Do not use controller tolerance to re-allow learner self-check in theory. Under v34, self-check questions belong to self_work/internal autocheck layers, not to theory.
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

VALIDATION TARGET MODE:
{target_mode}

PRIMARY STRUCTURED ARTIFACT KEYS:
{compact_json(primary_keys)}

PROMPT/SKILL FILES FOR CHECKED MATERIAL:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

SOURCE CONTRACT FROM JSON:
{compact_json(source_contract_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json(dependency_result_summaries_for_validation(spec, dependencies))}

STRUCTURED GENERATION ARTIFACTS FOR CHECKED MATERIAL:
{compact_json(validation_generation_artifacts)}

ARTIFACT VISIBILITY RULE:
- For current_control, current_control_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. Validator objections about missing visible keys should be overruled when this artifact is complete and internally consistent.
- The section above is structured generation output, not learner-facing HTML. It is the primary evidence for structured-content claims and may include teacher-only/internal fields.
- Do not uphold a validator claim of learner-facing key leakage merely because an internal artifact contains teacher-only data. In structured_artifacts mode, uphold leakage only if forbidden content appears in structured student-facing fields.
- For self_work, self_work_autocheck is the required non-rendered platform/QA layer for answers and autocheck settings. Validator objections about missing visible keys should be overruled when this artifact is complete and internally consistent.
- For intermediate, intermediate_assessment is the required non-rendered assessment/QA layer for keys, эталоны, rubrics, tests, and solutions. Validator objections about missing visible keys should be overruled when this artifact is complete and internally consistent.
- For mr_intermediate, dependency intermediate content/artifacts are source evidence only. They are not the checked material. Overrule claims that mr_intermediate duplicates a full KIM/variant bank unless the duplicated variants/tasks appear between CHECKED MATERIAL HTML START/END.

RULE VALIDATION:
{compact_json(validation_result_summary(rule_result))}

LLM VALIDATION:
{compact_json(validation_result_summary(llm_result))}

MERGED VALIDATION:
{compact_json(validation_result_summary(merged_validation))}

{html_section}
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
- For materials with primary_structured_artifact_keys, validate the structured generation_artifacts and do not semantically review rendered HTML. full_final_content is intentionally omitted for those materials because HTML can be regenerated from approved artifacts by a technical renderer.
- You receive full final HTML content only for HTML-first materials in FULL MATERIALRESULT OBJECTS. That content is not a preview and is not intentionally truncated.
- Do not claim that a material is truncated unless content_truncated is true or html_diagnostics proves an actual broken boundary in full_final_content.
- Validate only the current task package and the expected material specs listed below. Do not fail this lesson package because of future lessons, module-wide rows, or assessment content that was not generated in this task.
- Source JSON may contain known warnings such as difficulty.violation or module totals.raw percentages. If generated materials faithfully preserve and disclose such source warnings, report them as source_data_warnings, not as package-blocking material defects.
- Do not ask the generator to change lesson.practice_tasks counts or L1/L2 distribution when those values come from the source JSON. The generator must not silently rewrite source task composition to satisfy a norm.
- Approved individual materials remain approved at package validation time unless deterministic package rule issues prove missing materials, wrong order, or broken final files.
- Prefer approved=true with advisory issues when the materials are complete, importable HTML and individual material validation already approved them.

TASK JSON:
{compact_json(task)}

EXPECTED MATERIAL SPECS:
{compact_json([{"kind": spec.kind, "type": spec.material_type} for spec in specs])}

FULL MATERIALRESULT OBJECTS:
{compact_json([package_material_payload(item) for item in materials])}

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}
""".strip()
