# llama-routelog

Runs a JSONL chat session and records the two Qwen MoE routing tensors in the
Monet routing-log format.  Tool-specific arguments are `--session-script`,
`--output`, and `--session-id`; all remaining arguments are normal llama.cpp
arguments.  Input rows have `role` plus `content` or `content_file`.  An
`assistant` row requests greedy generation and may set `max_tokens`.

`--benchmark-no-capture` disables the eval callback for a paired overhead run;
that output is intentionally not a routing log and uses `benchmark.json`.
