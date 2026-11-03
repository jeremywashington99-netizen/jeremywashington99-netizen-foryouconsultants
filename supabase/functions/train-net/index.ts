// train-net: trains the engagement network inside Supabase and saves the run to public.model_runs.
//
// The caller must be a signed-in account (verify_jwt is on). Every database call below is made with
// THAT account's own token, so row-level security decides what it may see or write; this function
// holds no service key and no secret. The training code is js/net-core.js + js/mini-mlp.js from the
// public repo, pinned to one commit so the code that runs here cannot change without a redeploy.
import "https://raw.githubusercontent.com/jeremywashington99-netizen/jeremywashington99-netizen-foryouconsultants/606e915e30397480e9d50bce1c2e575ec1324bd1/js/dl-guardrails.js";
import "https://raw.githubusercontent.com/jeremywashington99-netizen/jeremywashington99-netizen-foryouconsultants/606e915e30397480e9d50bce1c2e575ec1324bd1/js/mini-mlp.js";
import "https://raw.githubusercontent.com/jeremywashington99-netizen/jeremywashington99-netizen-foryouconsultants/606e915e30397480e9d50bce1c2e575ec1324bd1/js/net-core.js";

// deno-lint-ignore no-explicit-any
const g: any = globalThis;
const CLASSES = ["Event setup", "Product / recipe", "POV hook", "Behind the scenes", "Announcement", "Static graphic"];
const URL_ = Deno.env.get("SUPABASE_URL")!;
const ANON = Deno.env.get("SUPABASE_ANON_KEY")!;
const MIN_GAP_MS = 15_000;       // one training run per client every 15 s: stops a button held down
const MAX_BODY = 2_000_000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "authorization, apikey, content-type, x-client-info",
  "access-control-allow-methods": "POST, OPTIONS",
};
const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { ...CORS, "content-type": "application/json" } });

function who(auth: string): { id: string } | null {
  try {
    const p = JSON.parse(atob(auth.replace(/^Bearer\s+/i, "").split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return p.role === "authenticated" && UUID.test(p.sub || "") ? { id: p.sub } : null;
  } catch { return null; }
}
const rest = (auth: string, path: string, init: RequestInit = {}) =>
  fetch(`${URL_}/rest/v1/${path}`, { ...init, headers: { apikey: ANON, authorization: auth, "content-type": "application/json", ...(init.headers || {}) } });

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (req.method !== "POST") return json(405, { ok: false, error: "POST only" });
  const auth = req.headers.get("authorization") || "";
  const user = who(auth);
  if (!user) return json(401, { ok: false, error: "sign in first" });
  if (Number(req.headers.get("content-length") || 0) > MAX_BODY) return json(413, { ok: false, error: "too much data in one request" });

  // deno-lint-ignore no-explicit-any
  let body: any;
  try { body = await req.json(); } catch { return json(400, { ok: false, error: "request was not valid JSON" }); }
  const clientId = body && body.client_id;
  if (!UUID.test(clientId || "")) return json(400, { ok: false, error: "client_id missing" });
  const bad = g.FYNetCore.validateClient(body.client);
  if (bad) return json(400, { ok: false, error: bad });

  // the client must be one this account owns
  const own = await rest(auth, `clients?id=eq.${clientId}&owner_id=eq.${user.id}&select=id`);
  if (!own.ok || !((await own.json()) || []).length) return json(403, { ok: false, error: "that client is not in your account" });

  // throttle
  const last = await rest(auth, `model_runs?client_id=eq.${clientId}&kind=eq.net&select=trained_at&order=trained_at.desc&limit=1`);
  if (last.ok) {
    const r = (await last.json()) || [];
    if (r[0] && Date.now() - new Date(r[0].trained_at).getTime() < MIN_GAP_MS) return json(429, { ok: false, error: "just trained; wait a few seconds" });
  }

  const t0 = performance.now();
  const c = body.client;
  const net = g.FYNetCore.create({ Mlp: g.FYMlp, G: g.FYGuard, classes: CLASSES });
  try {
    const st = await net.train(c);
    const row = await net.runRow(st, c, { owner_id: user.id, client_id: clientId });
    const m = st.metrics;
    const ins = await rest(auth, "model_runs?select=id,trained_at", { method: "POST", headers: { prefer: "return=representation" }, body: JSON.stringify(row) });
    if (!ins.ok) return json(502, { ok: false, error: "trained, but the database refused the save: " + (await ins.text()).slice(0, 200) });
    const saved = ((await ins.json()) || [])[0] || {};
    return json(200, { ok: true, run_id: saved.id, trained_at: saved.trained_at, rows: m.rows, epochs: m.epochs, ms: Math.round(performance.now() - t0) });
  } catch (e) {
    return json(422, { ok: false, error: String((e && (e as Error).message) || e) });
  }
});
