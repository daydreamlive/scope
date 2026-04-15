const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";

interface ImportSessionResponse {
  token: string;
  createUrl: string;
}

export async function createDaydreamImportSession(
  apiKey: string,
  workflowData: unknown,
  suggestedName?: string
): Promise<ImportSessionResponse> {
  const body: Record<string, unknown> = { workflowData };
  if (suggestedName) {
    body.suggestedName = suggestedName;
  }

  const response = await fetch(
    `${DAYDREAM_API_BASE}/v1/workflows/import-sessions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    }
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      `Failed to create import session: ${response.status} ${text}`
    );
  }

  return response.json() as Promise<ImportSessionResponse>;
}
