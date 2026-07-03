/**
 * Linkup web tools for Pi.
 *
 * Search is pinned to Linkup depth="fast" and outputType="searchResults" so lookups stay
 * low-latency and avoid Linkup's agentic/LLM modes. Fetch extracts clean markdown from a
 * single URL and can save it to disk to keep large pages out of model context.
 *
 * Env:
 *   LINKUP_API_KEY      required to register these tools
 *   LINKUP_SEARCH_URL   default https://api.linkup.so/v1/search
 *   LINKUP_FETCH_URL    default https://api.linkup.so/v1/fetch
 *   LINKUP_MAX_RESULTS  default 8
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";

const API_KEY = process.env.LINKUP_API_KEY || "";
const SEARCH_URL = process.env.LINKUP_SEARCH_URL || "https://api.linkup.so/v1/search";
const FETCH_URL = process.env.LINKUP_FETCH_URL || "https://api.linkup.so/v1/fetch";
const DEFAULT_MAX_RESULTS = clampInt(process.env.LINKUP_MAX_RESULTS, 8, 1, 20);
const DEFAULT_FETCH_RENDER_JS = envBool("LINKUP_FETCH_RENDER_JS", false);
const TOOL_TIMEOUT_MS = clampInt(process.env.LINKUP_TIMEOUT_MS, 60000, 1000, 180000);

function envBool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(raw.trim().toLowerCase());
}

function clampInt(raw: unknown, fallback: number, lo: number, hi: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, Math.trunc(n)));
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out = value.map((v) => String(v).trim()).filter(Boolean);
  return out.length ? out : undefined;
}

function insideCwd(out: string): string {
  const cwd = resolve(process.cwd());
  const target = isAbsolute(out) ? resolve(out) : resolve(cwd, out);
  if (!(target === cwd || target.startsWith(cwd + "/"))) {
    throw new Error("out must stay inside the current job working directory");
  }
  return target;
}

async function postJson(url: string, payload: Record<string, unknown>): Promise<any> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TOOL_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "authorization": `Bearer ${API_KEY}`,
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": "alexandria-searchbox-linkup/1.0",
      },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    const text = await res.text();
    let data: any;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { raw: text };
    }
    if (!res.ok) {
      throw new Error(`Linkup ${res.status}: ${JSON.stringify(data).slice(0, 500)}`);
    }
    return data;
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

export default function (pi: ExtensionAPI) {
  if (!API_KEY) return;

  pi.registerTool({
    name: "linkup_search",
    label: "Linkup Search",
    description:
      "Fast open-web search via Linkup. Always uses depth='fast' and outputType='searchResults' " +
      "for sub-second raw search results. Returns ranked {name,url,content,type} results. Use this " +
      "for current web context only when the dataroom is insufficient or the task asks for it.",
    parameters: Type.Object({
      query: Type.String({ description: "Natural-language search query." }),
      max_results: Type.Optional(Type.Number({ description: "Result cap, default 8, max 20." })),
      include_domains: Type.Optional(Type.Array(Type.String())),
      exclude_domains: Type.Optional(Type.Array(Type.String())),
      from_date: Type.Optional(Type.String({ description: "YYYY-MM-DD lower bound." })),
      to_date: Type.Optional(Type.String({ description: "YYYY-MM-DD upper bound." })),
      include_images: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, params) {
      const p = params as any;
      const payload: Record<string, unknown> = {
        q: String(p.query || ""),
        depth: "fast",
        outputType: "searchResults",
        maxResults: clampInt(p.max_results, DEFAULT_MAX_RESULTS, 1, 20),
      };
      const includeDomains = asStringArray(p.include_domains);
      const excludeDomains = asStringArray(p.exclude_domains);
      if (includeDomains) payload.includeDomains = includeDomains;
      if (excludeDomains) payload.excludeDomains = excludeDomains;
      if (p.from_date) payload.fromDate = String(p.from_date);
      if (p.to_date) payload.toDate = String(p.to_date);
      if (p.include_images !== undefined) payload.includeImages = Boolean(p.include_images);
      try {
        const data = await postJson(SEARCH_URL, payload);
        return ok(JSON.stringify({
          provider: "linkup",
          depth: "fast",
          outputType: "searchResults",
          query: payload.q,
          results: data.results || [],
        }, null, 2));
      } catch (e: any) {
        return err(String(e?.message || e));
      }
    },
  });

  pi.registerTool({
    name: "linkup_fetch",
    label: "Linkup Fetch",
    description:
      "Fetch one public URL as clean markdown via Linkup. Prefer passing out='sources/...' or " +
      "another job-local path so only a short preview enters context. Optional render_js is available " +
      "for client-rendered pages.",
    parameters: Type.Object({
      url: Type.String({ description: "Public HTTP/HTTPS URL to fetch." }),
      out: Type.Optional(Type.String({ description: "Optional output markdown path under the job cwd." })),
      render_js: Type.Optional(Type.Boolean()),
      include_raw_html: Type.Optional(Type.Boolean()),
      extract_images: Type.Optional(Type.Boolean()),
      max_chars: Type.Optional(Type.Number({ description: "Returned markdown cap when out is omitted." })),
    }),
    async execute(_id, params) {
      const p = params as any;
      const payload = {
        url: String(p.url || ""),
        renderJs: p.render_js === undefined ? DEFAULT_FETCH_RENDER_JS : Boolean(p.render_js),
        includeRawHtml: Boolean(p.include_raw_html),
        extractImages: Boolean(p.extract_images),
      };
      try {
        const data = await postJson(FETCH_URL, payload);
        const markdown = String(data.markdown || "");
        const images = Array.isArray(data.images) ? data.images : [];
        if (p.out) {
          const target = insideCwd(String(p.out));
          mkdirSync(dirname(target), { recursive: true });
          writeFileSync(target, markdown);
          return ok(JSON.stringify({
            provider: "linkup",
            url: payload.url,
            path: target,
            chars: markdown.length,
            images: images.length,
            preview: markdown.slice(0, 1200),
          }, null, 2));
        }
        const cap = clampInt(p.max_chars, 20000, 1000, 100000);
        return ok(JSON.stringify({
          provider: "linkup",
          url: payload.url,
          markdown: markdown.slice(0, cap),
          truncated: markdown.length > cap,
          chars: markdown.length,
          images,
          rawHtml: payload.includeRawHtml ? data.rawHtml : undefined,
        }, null, 2));
      } catch (e: any) {
        return err(String(e?.message || e));
      }
    },
  });
}
