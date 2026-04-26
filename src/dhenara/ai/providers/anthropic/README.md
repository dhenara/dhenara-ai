# Anthropic Provider Notes

This provider supports two Anthropic structured-output paths:

- Claude 4.6 and newer use Anthropic native structured output via `output_config.format`.
- Older Anthropic models fall back to tool-based structured output.

Tools and structured output are separate concerns for the native path. The provider keeps
native structured output enabled when actual tools are present.

Anthropic still rejects requests that combine all of the following:

- `thinking` enabled
- tools present
- forced tool use via `tool_choice` (`any` or `tool`)

The observed API error is:

`Thinking may not be enabled when tool_choice forces tool use.`

Because of that current API contract, the provider relaxes forced tool choice to
`{"type": "auto"}` when thinking is enabled. This is not a structured-output fallback.
It preserves native structured output for Claude 4.6+ and only adjusts tool-choice
strictness so the request remains valid.

If Anthropic removes this API restriction in a future release, the corresponding logic in
`chat.py` and the DAI-117 / DAI-118 regression tests can be revisited.

API Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use