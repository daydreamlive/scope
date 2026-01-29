/**
 * CustomWebComponent - Loads and renders custom Web Components from external scripts.
 *
 * This enables plugins/pipelines to inject their own UI by:
 * 1. Serving a JS file that defines a Web Component (Custom Element)
 * 2. Specifying script_url and element_name in ui_field_config
 *
 * The custom element receives:
 * - data-value: Current field value (JSON stringified)
 * - data-config: Additional config from schema (JSON stringified)
 * - data-disabled: Whether the field is disabled
 *
 * The custom element should emit a "change" CustomEvent with detail.value
 * when the value changes.
 */

import { useEffect, useRef, useState } from "react";

// Track loaded scripts to avoid duplicate loading
const loadedScripts = new Set<string>();
const loadingPromises = new Map<string, Promise<void>>();

/**
 * Load a script from URL, caching to avoid duplicate loads.
 */
async function loadScript(url: string): Promise<void> {
  if (loadedScripts.has(url)) {
    return;
  }

  // Check if already loading
  const existing = loadingPromises.get(url);
  if (existing) {
    return existing;
  }

  // Load the script
  const promise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.type = "module";
    script.onload = () => {
      console.log(`[CustomWebComponent] Script onload fired: ${url}`);
      loadedScripts.add(url);
      loadingPromises.delete(url);
      // Give the module time to execute after loading
      setTimeout(resolve, 100);
    };
    script.onerror = event => {
      console.error(`[CustomWebComponent] Script onerror fired:`, event);
      loadingPromises.delete(url);
      reject(new Error(`Failed to load script: ${url}`));
    };
    document.head.appendChild(script);
  });

  loadingPromises.set(url, promise);
  return promise;
}

export interface CustomWebComponentProps {
  /** URL to load the web component script from */
  scriptUrl: string;
  /** Custom element tag name (e.g., "image-filter-layers") */
  elementName: string;
  /** Current field value */
  value: unknown;
  /** Called when the custom element emits a change event */
  onChange: (value: unknown) => void;
  /** Additional config to pass to the element */
  config?: Record<string, unknown>;
  /** Whether the field is disabled */
  disabled?: boolean;
  /** Field key for identification */
  fieldKey: string;
}

export function CustomWebComponent({
  scriptUrl,
  elementName,
  value,
  onChange,
  config,
  disabled = false,
  fieldKey,
}: CustomWebComponentProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const elementRef = useRef<HTMLElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Use a ref for onChange to avoid re-running the main effect when it changes
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // Load script and create element - only depends on stable values
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        setLoading(true);
        setError(null);

        // First check if element is already defined (from a previous load)
        const alreadyDefined = customElements.get(elementName);
        if (alreadyDefined) {
          console.log(
            `[CustomWebComponent] Element "${elementName}" already defined, creating instance`
          );
        } else {
          console.log(`[CustomWebComponent] Loading script: ${scriptUrl}`);

          // Load the script
          await loadScript(scriptUrl);

          console.log(
            `[CustomWebComponent] Script loaded, checking for element: ${elementName}`
          );

          if (!mounted) {
            console.log(
              `[CustomWebComponent] Component unmounted during script load, aborting`
            );
            return;
          }

          // Check if the custom element is defined
          const isDefined = customElements.get(elementName);
          if (!isDefined) {
            console.log(
              `[CustomWebComponent] Element not yet defined, waiting...`
            );
            // Wait for the element to be defined with a timeout
            const timeoutMs = 10000;
            const timeoutPromise = new Promise<never>((_, reject) => {
              setTimeout(() => {
                reject(
                  new Error(
                    `Custom element "${elementName}" was not defined within ${timeoutMs}ms. Check browser console for script errors.`
                  )
                );
              }, timeoutMs);
            });

            await Promise.race([
              customElements.whenDefined(elementName),
              timeoutPromise,
            ]);
          }
        }

        if (!mounted) {
          console.log(`[CustomWebComponent] Component unmounted, aborting`);
          return;
        }

        if (!containerRef.current) {
          console.log(`[CustomWebComponent] Container ref not available`);
          return;
        }

        console.log(
          `[CustomWebComponent] Creating element instance: ${elementName}`
        );

        // Create the custom element
        const element = document.createElement(elementName);
        element.setAttribute("data-field-key", fieldKey);
        // If value is already a string (e.g., JSON stored as string in schema), use it directly
        // Otherwise, JSON.stringify it. Handle undefined/null gracefully.
        let valueStr: string;
        if (value === undefined || value === null) {
          valueStr = "{}";
        } else if (typeof value === "string") {
          valueStr = value;
        } else {
          valueStr = JSON.stringify(value);
        }
        element.setAttribute("data-value", valueStr);
        element.setAttribute("data-config", JSON.stringify(config ?? {}));
        element.setAttribute("data-disabled", String(disabled));

        // Listen for change events - use the ref to get the current callback
        const handleChange = (event: Event) => {
          const customEvent = event as CustomEvent;
          if (customEvent.detail?.value !== undefined) {
            onChangeRef.current(customEvent.detail.value);
          }
        };
        element.addEventListener("change", handleChange);

        // Store the element ref first, then update loading state
        // React will handle removing the loading message when state changes
        elementRef.current = element;

        // Set loading to false first - this will cause React to re-render
        // and remove the loading message
        setLoading(false);

        console.log(`[CustomWebComponent] Component loaded successfully`);
      } catch (err) {
        console.error(`[CustomWebComponent] Error:`, err);
        if (mounted) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      mounted = false;
    };
    // Only re-run when scriptUrl, elementName, or fieldKey changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scriptUrl, elementName, fieldKey]);

  // Append the element to the container after loading completes
  useEffect(() => {
    if (!loading && !error && elementRef.current && containerRef.current) {
      // Only append if not already in the DOM
      if (!elementRef.current.parentNode) {
        containerRef.current.appendChild(elementRef.current);
      }
    }
  }, [loading, error]);

  // Update element attributes when props change
  useEffect(() => {
    if (elementRef.current) {
      let valueStr: string;
      if (value === undefined || value === null) {
        valueStr = "{}";
      } else if (typeof value === "string") {
        valueStr = value;
      } else {
        valueStr = JSON.stringify(value);
      }
      elementRef.current.setAttribute("data-value", valueStr);
    }
  }, [value]);

  useEffect(() => {
    if (elementRef.current) {
      elementRef.current.setAttribute(
        "data-config",
        JSON.stringify(config ?? {})
      );
    }
  }, [config]);

  useEffect(() => {
    if (elementRef.current) {
      elementRef.current.setAttribute("data-disabled", String(disabled));
    }
  }, [disabled]);

  // Always render the container so the ref is available during effect
  return (
    <div ref={containerRef} className="custom-web-component">
      {loading && (
        <div className="flex items-center justify-center p-4 text-sm text-muted-foreground">
          Loading component...
        </div>
      )}
      {error && (
        <div className="p-4 text-sm text-destructive border border-destructive/50 rounded-md">
          Failed to load custom component: {error}
        </div>
      )}
    </div>
  );
}
