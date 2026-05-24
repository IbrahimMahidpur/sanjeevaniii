import { useEffect, useState, useRef } from "react";

export function useSSEStream(
  url: string,
  params: Record<string, string | undefined> | null
) {
  const [data, setData] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<boolean>(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!params) return;

    setData("");
    setDone(false);
    setError(null);

    // Cancel previous request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const qs = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v as string)}`)
      .join("&");

    const fullUrl = `${url}?${qs}`;

    async function fetchStream() {
      try {
        const response = await fetch(fullUrl, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });

        if (!response.ok) {
          setError(`HTTP error: ${response.status}`);
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) { setError("No response body"); return; }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done: streamDone } = await reader.read();
          if (streamDone) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith("data:")) continue;

            const raw = trimmed.slice(5).trim();
            if (raw === "[DONE]") { setDone(true); reader.cancel(); return; }

            try {
              const payload = JSON.parse(raw);
              if (payload.token !== undefined) {
                setData((prev) => prev + payload.token);
              }
              if (payload.done) { setDone(true); reader.cancel(); return; }
            } catch { /* ignore non-JSON */ }
          }
        }
        setDone(true);
      } catch (err: any) {
        if (err.name !== "AbortError") {
          setError(`Stream error: ${err.message}`);
        }
      }
    }

    fetchStream();

    return () => { controller.abort(); };
  }, [url, JSON.stringify(params)]);

  return { data, error, done };
}