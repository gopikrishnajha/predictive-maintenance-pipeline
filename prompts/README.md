# Prompts Directory

This directory contains use-case-specific prompt configurations for the predictive_maintenance_pipeline multi-agent system.

## File Structure

Each use case has its own prompt file with all prompts for that use case in a single file:
- `pipeline_defects_detection.txt` - Pipeline infrastructure defect detection
- `solar_panel_defects_detection.txt` - Solar panel defect detection (template)

## Prompt File Format

Prompts are organized using section markers:

```
# Comments start with # and are ignored

[SECTION_NAME]
Your prompt text here.
Can span multiple lines.

[ANOTHER_SECTION]
Another prompt here.
```

### Standard Sections

Each use case file should contain these sections:
- `[POLICY]` - Prompt for policy/filtering rules
- `[ANALYSIS]` - Prompt for data analysis and summarization
- `[EVIDENCE]` - Prompt for audit trail generation

### Adding Custom Sections

You can add custom sections as needed:
```
[CUSTOM_SECTION]
Your custom prompt text...
```

Then access it in code:
```python
from src.prompt_loader import load_prompts
prompts = load_prompts("prompts/your_usecase.txt")
custom_prompt = prompts["custom_section"]  # lowercase
```

## Usage in Code

### Load All Prompts
```python
from src.prompt_loader import load_prompts

prompts = load_prompts("prompts/pipeline_defects_detection.txt")
policy_prompt = prompts["policy"]
analysis_prompt = prompts["analysis"]
```

### Load Single Section
```python
from src.prompt_loader import load_prompt_section

policy = load_prompt_section("prompts/pipeline_defects_detection.txt", "policy")
```

### Command Line
```bash
# Use default use case (pipeline_defects_detection)
python scripts/query_vdms_agents.py --limit 30

# Use different use case
python scripts/query_vdms_agents.py --use-case solar_panel_defects_detection --limit 30
```

## Adding a New Use Case

1. Create a new prompt file: `prompts/my_new_usecase.txt`
2. Add sections using the format above
3. Run with: `python scripts/run_agent_orchestration.py --use-case my_new_usecase`

Example template:
```
# My New Use Case - Prompt Configuration

[POLICY]
Your policy rules here...

[ANALYSIS]
Your analysis instructions here...

[EVIDENCE]
Your audit trail requirements here...
```

## Best Practices

1. **Use descriptive comments** to explain complex prompts
2. **Keep sections focused** - each section should have one clear purpose
3. **Version control** - commit prompt changes with clear messages
4. **Test prompts** in both fallback and model modes
5. **Document requirements** - note any special formatting or constraints
