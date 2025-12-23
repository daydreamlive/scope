# Contributing

Thank you for your interest in contributing!

We use the [Developer Certificate of Origin (DCO)](https://developercertificate.org/) to ensure that all contributions can be legally included and relicensed in future releases.

To certify your contribution, please sign off each commit with:

    git commit -s

This adds a line like this to your commit message:

    Signed-off-by: Your Name <your.email@example.com>

By signing off, you confirm that you have the right to submit this code and that it may be distributed under the same license as this project.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) which is needed to run the server, [Node.js](https://nodejs.org/en/download) which is needed to build frontend and [ffmpeg](https://www.ffmpeg.org/download.html) which is needed for certain media workflows during development.

## Frontend

First, navigate to the `frontend` directory.

Install dependencies:

```bash
npm install
```

Run the development server (hot reloading is on by default):

```bash
npm run dev
```

## Server

Install all (including development) dependencies:

```bash
uv sync --group dev
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run the server with hot reloading enabled:

```bash
uv run daydream-scope --reload
```

## Server Tests

```bash
uv run pytest
```

## Firewalls

See [firewalls section](https://github.com/daydreamlive/scope?tab=readme-ov-file#firewalls) of the README.

If you have SSH access to the remote machine, you can also setup SSH port forwarding

If you are using Cursor, it will automatically do this for you if you connect to the remote machine within the editor and run the Scope server from within the Cursor terminal.

You can also manually update your SSH config file eg `~/.ssh/config` to look like this:

```
Host <NAME>
  Hostname <IP>
  User <USER>
  Port <PORT>
  IdentityFile <IDENTITY_FILE>
  PreferredAuthentications publickey
  LocalForward 8000 127.0.0.1:8000
```

- You can set `Host` to a nickname for this machine.
- You should set `Hostname`, `User` and `Port` based on the SSH information for the machine.
- You should set `IdentityFile` to the path to the file containing your SSH private key.

## Testing Pipelines

By default, the server does not load any pipelines on startup, but you can set the `PIPELINE` environment variable to automatically load a specific pipeline on startup which can be useful for testing.

This would load the `longlive` pipeline on startup:

```bash
PIPELINE="longlive" uv run daydream-scope
```

You can also test the `longlive` pipeline on its own:

```bash
uv run -m scope.core.pipelines.longlive.test
```

This test outputs a video file.

## Release Process

1. Update the `version` field in `package.json` for the frontend.
2. Update the `version` field in `pyproject.toml` for the server.
