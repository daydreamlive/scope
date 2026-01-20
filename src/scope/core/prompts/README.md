# Prompts

`interactive_example.jsonl` contains example prompt sequences for interactive generation from [MemFlow](https://github.com/KlingTeam/MemFlow/blob/main/prompts/interactive_example.jsonl).

A pipeline test script can be run using each of the prompt sequences like this:

```
uv run -m scope.core.pipelines.memflow.test --prompts src/core/prompts/interactive_example.jsonl
```
