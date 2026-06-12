def build_r3_judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return f"""Evaluate the response based on the given task, input, two responses, and evaluation rubric.
Provide a fair and detailed assessment following the rubric.

### TASK
Determine which assistant response is better overall for the given user input.

### INPUT
{question}

### RESPONSE 1
{answer_a}

### RESPONSE 2
{answer_b}

### EVALUATION RUBRIC
Response 1: Response 1 provided better response, rejecting Response 2.
Response 2: Response 2 provided better response, rejecting Response 1.

### OUTPUT FORMAT
Return a JSON response in the following format:

{{
"explanation": "Explanation of why one response is preferred over the other",
"score": "Final selection between 'Response 1' or 'Response 2'"
}}

### EVALUATION
"""

def build_r3_judge_prompt_unbiased(question: str, answer_a: str, answer_b: str) -> str:
    return f"""You are an impartial evaluator.

Your job: decide which response better satisfies the user's input.
The two responses are anonymized; their order is arbitrary. Do NOT reward or penalize a response for being Response 1 vs Response 2.

### TASK
Determine which assistant response is better overall for the given user input.

### INPUT
{question}

### RESPONSE 1
{answer_a}

### RESPONSE 2
{answer_b}

### EVALUATION PRINCIPLES (follow strictly)
- Be unbiased: ignore position/order effects.
- Evaluate each response independently against the INPUT.
- Prefer responses that are: correct, complete, directly responsive, well-reasoned, and appropriately safe.
- Penalize: factual errors, hallucinations, missing key requirements, unsafe guidance, refusal when not needed, excessive verbosity, and irrelevant content.
- If the user asked for a specific format or constraints, compliance is mandatory.

### SCORING CRITERIA
Consider (in this order of importance):
1) Correctness & factual reliability
2) Instruction-following (including required format/constraints)
3) Completeness & coverage of key points
4) Clarity & organization
5) Safety, ethics, and policy compliance

### TIE-BREAKERS (use only if extremely close)
A) Better matches the user's explicit constraints and intent
B) Fewer unsupported claims / less speculation
C) More concise while still fully answering
D) If still indistinguishable after A–C, pick the one with clearer reasoning

### OUTPUT FORMAT (MUST FOLLOW EXACTLY)
Return ONLY valid JSON (no markdown, no extra text) with exactly these keys:

{{
  "explanation": "<clear, specific justification referencing the criteria above>",
  "score": "<exactly 'Response 1' or 'Response 2'>"
}}

Rules:
- The value of "score" MUST be exactly one of: Response 1, Response 2.
- Do not add any other keys.
- Do not include trailing commentary outside the JSON.

### EVALUATION
"""