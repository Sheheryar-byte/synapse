"""synapse CLI — synapse new <name> and synapse run <file>."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="synapse",
    help="Synapse AI Runtime CLI",
    add_completion=False,
)

_AGENT_TEMPLATE = '''\
from synapse import agent, tool, workflow, step

@tool
def my_tool(query: str) -> str:
    """A placeholder tool. Replace with real logic."""
    return f"Result for: {query}"

@agent(model="gpt-4o")
async def assistant(prompt: str) -> str:
    """Main agent. Edit model, add tools, adjust as needed."""
    ...

@workflow
async def main_pipeline(prompt: str) -> str:
    return await step(assistant, prompt)
'''

_ENV_TEMPLATE = """\
# Copy this file to .env and fill in your keys.
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...
# OLLAMA_HOST=http://localhost:11434
"""


@app.command()
def new(
    name: str = typer.Argument(..., help="Project name / directory to create"),
) -> None:
    """Scaffold a new Synapse project."""
    project_dir = Path(name)
    if project_dir.exists():
        typer.echo(f"Error: '{name}' already exists.", err=True)
        raise typer.Exit(1)

    project_dir.mkdir(parents=True)
    (project_dir / "agent.py").write_text(_AGENT_TEMPLATE)
    (project_dir / ".env.example").write_text(_ENV_TEMPLATE)
    (project_dir / "requirements.txt").write_text("synapse-runtime\n")

    typer.echo(f"✓ Created project '{name}/'")
    typer.echo(f"  {name}/agent.py       — your first agent")
    typer.echo(f"  {name}/.env.example   — copy to .env and add your API keys")
    typer.echo(f"\nNext steps:")
    typer.echo(f"  cd {name}")
    typer.echo(f"  cp .env.example .env  # add your keys")
    typer.echo(f"  synapse run agent.py")


@app.command()
def run(
    file: Path = typer.Argument(..., help="Python file to run (e.g. agent.py)"),
) -> None:
    """Run a Synapse agent file."""
    if not file.exists():
        typer.echo(f"Error: '{file}' not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"▶ Running {file} …")
    result = subprocess.run([sys.executable, str(file)])
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
