/**
 * Local-only tools over the uploaded dataroom, served by server/dataroom_service.py.
 *
 * DEFAULT (loaded when SEARCHBOX_TOOLS is unset) - two ATOMIC model primitives:
 *   sentence_embed  -> jina-embeddings-v5-text-small  (text -> vector, written to jsonl)
 *   passage_rerank  -> jina-reranker-v3               (query + docs -> relevance scores)
 * These are deliberately raw single-model operations: the model itself decides where passages
 * come from (grep/read/chunking) and how to use the embeddings (similarity/clustering).
 *
 * OPT-IN (only loaded when SEARCHBOX_TOOLS explicitly lists it):
 *   semantic_search -> the high-level embed->rank->top-k pipeline (kept in the repo but NOT
 *                      loaded by default, so we can observe whether the model assembles its own
 *                      retrieval from the atomic sentence_embed primitive).
 *
 * Tools are gated by SEARCHBOX_TOOLS so the ablation harness can pick the set without editing
 * this file. DATAROOM_INDEX_URL defaults to http://127.0.0.1:8078.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const BASE = process.env.DATAROOM_INDEX_URL || "http://127.0.0.1:8078";
const TOOL_TIMEOUT_MS = Number(process.env.DATAROOM_TOOL_TIMEOUT_MS || 900000);

// Tool catalog: single source of truth (pi/tools-catalog.json). ~20 tools, every one a thin
// wrapper over the embedding/reranking backend in dataroom_service.py.
const CATALOG_PATH = join(dirname(dirname(fileURLToPath(import.meta.url))), "tools-catalog.json");
type ToolSpec = {
  name: string; op: string; desc: string; default?: boolean;
  params: Record<string, string>;
  fixed?: Record<string, unknown>;
  rename?: Record<string, string>;
};
const CATALOG: ToolSpec[] = (() => {
  try {
    return JSON.parse(readFileSync(CATALOG_PATH, "utf8")).tools as ToolSpec[];
  } catch {
    return [];
  }
})();

// Ablation gate: which tools this run is allowed to register.
//   SEARCHBOX_TOOLS UNSET (env var absent) => the DEFAULT set (see DEFAULT_TOOLS below).
//   SEARCHBOX_TOOLS=""   (present but empty) => NO external tools (pi built-ins only).
//   SEARCHBOX_TOOLS="a,b"                    => exactly those tools.
const ENABLED = (() => {
  const rawEnv = process.env.SEARCHBOX_TOOLS;
  if (rawEnv === undefined) return null; // unset => use DEFAULT_TOOLS
  return new Set(rawEnv.trim().split(",").map((s) => s.trim()).filter(Boolean));
})();
// Backward-compatible aliases so older SEARCHBOX_TOOLS values keep working.
const ALIAS: Record<string, string> = {
  dataroom_search: "sentence_embed",
  dataroom_rerank: "passage_rerank",
};
// The default tool set when SEARCHBOX_TOOLS is unset = the catalog entries flagged default:true
// (sentence_embed + passage_rerank). All other tools are opt-in via SEARCHBOX_TOOLS.
const DEFAULT_TOOLS = new Set(CATALOG.filter((t) => t.default).map((t) => t.name));
const enabled = (name: string) => {
  if (ENABLED === null) return DEFAULT_TOOLS.has(name);
  return ENABLED.has(name) || [...ENABLED].some((n) => ALIAS[n] === name);
};

async function call(op: string, body: Record<string, unknown>): Promise<string> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TOOL_TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}/${op}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`dataroom service ${op} -> ${res.status}: ${text}`);
    return text;
  } catch (e: any) {
    const msg = String(e?.message || e);
    throw new Error(`dataroom service ${op} fetch failed after ${TOOL_TIMEOUT_MS}ms: ${msg}`);
  } finally {
    clearTimeout(timer);
  }
}

function ok(text: string) {
  return { content: [{ type: "text" as const, text }], details: {} };
}
function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true, details: {} };
}


// Build a typebox parameter schema from a compact param spec ("s","s[]","i?","n?","s[]?", ...).
function buildParams(spec: Record<string, string>) {
  const props: Record<string, any> = {};
  for (const [key, t] of Object.entries(spec)) {
    const optional = t.endsWith("?");
    const base = optional ? t.slice(0, -1) : t;
    let schema: any;
    if (base === "s") schema = Type.String();
    else if (base === "s[]") schema = Type.Array(Type.String());
    else if (base === "i" || base === "n" || base === "f") schema = Type.Number();
    else schema = Type.String();
    props[key] = optional ? Type.Optional(schema) : schema;
  }
  return Type.Object(props);
}

// Coerce/validate one tool's params into the backend request body, honoring rename + fixed.
function buildBody(spec: ToolSpec, p: any): Record<string, unknown> | string {
  const body: Record<string, unknown> = {};
  for (const [key, t] of Object.entries(spec.params)) {
    const optional = t.endsWith("?");
    const base = optional ? t.slice(0, -1) : t;
    let v = p[key];
    if (v === undefined || v === null || (typeof v === "string" && v === "")) {
      if (!optional) return `${key} is required`;
      continue;
    }
    if (base === "s[]") {
      if (!Array.isArray(v)) return `${key} must be an array of strings`;
      v = v.filter((x: any) => typeof x === "string");
      if (!v.length && !optional) return `${key} must be a non-empty array of strings`;
    } else if (base === "i" || base === "n" || base === "f") {
      if (!Number.isFinite(Number(v))) { if (!optional) return `${key} must be a number`; continue; }
      v = base === "i" ? Math.trunc(Number(v)) : Number(v);
    } else {
      v = String(v);
    }
    // rename: map this arg to a different backend key, with a few structural transforms
    const target = (spec.rename && spec.rename[key]) || key;
    if (target === "texts_one") body["texts"] = [v];        // single string -> texts:[v]
    else if (target === "queries_one") body["queries"] = [v];
    else body[target] = v;
  }
  if (spec.fixed) Object.assign(body, spec.fixed);
  return body;
}

export default function (pi: ExtensionAPI) {
  for (const spec of CATALOG) {
    if (!enabled(spec.name)) continue;
    pi.registerTool({
      name: spec.name,
      label: spec.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      description: spec.desc,
      parameters: buildParams(spec.params),
      async execute(_id, params) {
        const body = buildBody(spec, params as any);
        if (typeof body === "string") return err(body);
        try {
          return ok(await call(spec.op, body));
        } catch (e: any) {
          return err(String(e?.message || e));
        }
      },
    });
  }
}
