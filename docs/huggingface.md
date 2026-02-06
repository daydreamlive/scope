# HuggingFace Authentication

Some pipelines use gated models on HuggingFace that require authentication to download. A HuggingFace token is also used to obtain Cloudflare TURN credentials for WebRTC connections behind firewalls. You can create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Scope supports two ways to provide your token: through the Settings UI or via an environment variable.

## Using the API Keys Tab

1. Click the gear button in the app header to open the Settings dialog.

2. Navigate to the **API Keys** tab.

3. Find the **HuggingFace** entry.

4. Paste your token into the field and click the save button.

5. To remove a stored token, click the delete button next to the entry.

> **Note:** If the `HF_TOKEN` environment variable is already set, the input field is disabled.

## Using an Environment Variable

Setting the `HF_TOKEN` environment variable is the preferred method for headless, cloud, and CI deployments.

**Linux / macOS:**

```bash
export HF_TOKEN=hf_your_token_here
```

**Windows (PowerShell):**

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

> **Note:** The environment variable takes precedence over a token stored through the UI.

## When Is a Token Needed?

- **Gated model downloads** — Pipelines that depend on gated HuggingFace models will fail with an authentication error without a valid token.
- **Cloudflare TURN (WebRTC)** — A token is used to obtain TURN server credentials for NAT traversal. Without it, Scope falls back to a public STUN server which may not work behind strict firewalls.

## Troubleshooting

- **Authentication errors during model download** — Verify that your token is set in Settings > API Keys or via the `HF_TOKEN` environment variable.
- **Token is set but downloads still fail** — Ensure you have accepted the model's license agreement on its HuggingFace page. Gated models require explicit approval before access is granted.
