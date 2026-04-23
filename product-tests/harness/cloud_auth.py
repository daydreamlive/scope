"""Cloud auth bypass — pre-seed localStorage so the app skips the sign-in
redirect during cloud-mode onboarding tests.

Production flow: user clicks Sign In in CloudAuthStep, is redirected to
``app.daydream.live/sign-in/local``, completes OAuth, returns with a token
that the app stores at ``localStorage["daydream_auth"]``.

Test flow: we inject the same-shaped blob via ``addInitScript`` so the
``isAuthenticated()`` check in ``frontend/src/lib/auth.ts:212`` sees a
valid key+user and auto-advances past the cloud_auth phase.

The injected key is recognizable (``test-bypass-*``) so backend logs show
clearly which requests came from test bypass. Real API calls that hit the
Daydream auth backend with this key will fail — tests that rely on the
bypass must also have the backend's cloud relay pointed at a fal app that
doesn't enforce auth, or ``SCOPE_CLOUD_AUTH_BYPASS=1`` set so the backend
short-circuits the auth check.
"""

from __future__ import annotations

import json
import time

from playwright.sync_api import BrowserContext

AUTH_STORAGE_KEY = "daydream_auth"


def make_test_auth_blob() -> dict:
    """Return a daydream_auth-shaped dict for localStorage injection."""
    return {
        "apiKey": f"test-bypass-{int(time.time())}",
        "userId": "test-user-0000",
        "displayName": "Test Bypass User",
        "email": "test-bypass@daydream.live",
        "cohortParticipant": False,
        "isAdmin": False,
    }


def install_cloud_auth_bypass(
    context: BrowserContext, blob: dict | None = None
) -> dict:
    """Install an init script that pre-seeds localStorage before the app loads.

    Must be called BEFORE the first ``page.goto()``. Returns the auth blob
    that was injected (so tests can assert on it if needed).
    """
    blob = blob or make_test_auth_blob()
    payload = json.dumps(blob)
    script = f"""
        try {{
            window.localStorage.setItem({json.dumps(AUTH_STORAGE_KEY)}, {json.dumps(payload)});
        }} catch (e) {{
            console.warn('[product-tests] failed to seed auth:', e);
        }}
    """
    context.add_init_script(script)
    return blob
