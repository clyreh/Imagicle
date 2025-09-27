// Minimal API client stubs
export async function ping(base="/api"){ const r=await fetch(`${base}/healthz`); return r.json(); }
