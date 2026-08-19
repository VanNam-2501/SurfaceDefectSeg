import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the aluminum segmentation product", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Aluminum Surface Lab \| Segmentation Demo<\/title>/i);
  assert.match(html, /Compare three defect-segmentation models on one image\./);
  assert.match(html, /U-Net \/ ResNet-18/);
  assert.match(html, /SegFormer-B0/);
  assert.match(html, /VMamba-T/);
  assert.match(html, /type="file"/);
  assert.match(html, /Fully automatic hybrid policy active|Frozen spatial policy active|Decision policy missing/);
  assert.match(html, /PASS \/ REVIEW \/ DEFECT/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("client connects health status to model availability and inference", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /fetch\(`\$\{apiBase\}\/health`\)/);
  assert.match(page, /nextStatus\[key\]\?\.available/);
  assert.match(page, /modelState\.policy_compatible === false/);
  assert.match(page, /fetch\(`\$\{apiBase\}\/infer`, \{ method: "POST", body \}\)/);
  assert.match(page, /availableCount !== selected\.length/);
  assert.match(page, /setDecision\(data\.decision/);
});
