---
sidebar_label: Installation
---

# Installation

Get Wren AI running. Your AI coding agent does the rest.

## 1. Install the skill

This installs a single **discovery stub** (`wren`) that teaches AI coding agents (Claude Code, Openclaw, Hermes, Codex, etc.) how to drive the Wren CLI for you:

```bash
npx skills add Canner/WrenAI
```

Have multiple AI coding agents installed and want the stub available in all of them? Pass `--agent '*'`:

```bash
npx skills add Canner/WrenAI --agent '*'
```

Or via the install script:

```bash
curl -fsSL https://raw.githubusercontent.com/Canner/WrenAI/main/skills/install.sh | bash
```

> **Only one skill is installed** — `wren` (at `~/.claude/skills/wren/SKILL.md` for Claude Code). This is expected. Since Wren `0.8`, the workflow guides (`onboarding`, `usage`, `generate-mdl`, `dlt-connector`, `enrich-context`) no longer install as separate skills; they live inside the `wren` CLI and the stub fetches them on demand with `wren skills get <name>`. See the [Skills reference](/oss/reference/skills) for the full delivery model and what each guide does.

## 2. Ask your agent to set things up

**Start a new agent session** (skills load at session start), open your project directory, and ask:

Use the `/wren` skill to install and set up Wren AI.

The agent will check your environment, install Python dependencies, create a connection profile for your data source, scaffold the project, and run a first query — all in one flow.

## 3. Start asking questions

Once onboarding finishes, just ask your agent business questions in natural language. The agent uses Wren AI's context layer to resolve schema, recall similar past queries, and generate accurate SQL.

```text
How many customers placed more than one order this month?
```

```text
What are the top 5 products by total revenue?
```

## Optional: call the model through the API

If your installed coding-agent CLI does not expose the model you want, Wren
can call the OpenAI API directly. The API backend defaults to `gpt-4o-mini` and
uses the same Wren context/query workflow when run inside a Wren project:

```bash
wren ai auth login
wren ai use openai --model gpt-4o-mini
wren ask "What are the top 5 products by total revenue?" --guided
```

Switch back at any time:

```bash
wren ai use prompt   # return to the existing external-agent prompt flow
wren ai use codex    # use the locally authenticated Codex CLI
```

`wren ai auth login` stores `OPENAI_API_KEY` in `~/.wren/.env`. This API key is
separate from `codex login`; a ChatGPT/Codex login cannot be reused as an API
key.

## What's next

- [Quickstart](./quickstart.md) — walk through a full example with the bundled `jaffle_shop` sample dataset
- [Connect your database](/oss/guides/connect) — connect a profile to a real data source
- [Skills reference](/oss/reference/skills) — what each skill does in detail
