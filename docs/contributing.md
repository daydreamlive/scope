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

## Testing Pipelines

By default, the server does not load any pipelines on startup, but you can set the `PIPELINE` environment variable to automatically load a specific pipeline on startup which can be useful for testing.

This would load the `longlive` pipeline on startup:

```bash
PIPELINE="longlive" uv run daydream-scope
```

You can also test the `longlive` pipeline on its own:

```bash
uv run -m pipelines.longlive.test
```

This test outputs a video file.

## Release Process

TBD
