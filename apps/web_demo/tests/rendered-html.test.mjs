import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", String(process.pid) + "-" + String(Date.now()));
  const { default: handler } = await import(workerUrl.href);

  return handler(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
  );
}

test("server-renders raw baselines and all calibrated choices", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Aluminum Surface Lab \| Segmentation Demo<\/title>/i);
  assert.match(html, /Compare original model outputs with calibrated rule-based decisions\./);
  assert.match(html, /Original U-Net/);
  assert.match(html, /Original SegFormer/);
  assert.match(html, /Original VMamba/);
  assert.match(html, /U-Net rule-based/);
  assert.match(html, /SegFormer rule-based/);
  assert.match(html, /VMamba rule-based/);
  assert.match(html, /U-Net \+ SegFormer/);
  assert.match(html, /U-Net \+ VMamba/);
  assert.match(html, /SegFormer \+ VMamba/);
  assert.match(html, /type="file"/);
  assert.match(html, /type="radio"/);
  assert.match(html, /Selected mode unavailable|policy ready|baseline ready/);
  assert.match(html, /PASS \/ REVIEW \/ DEFECT/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("client connects health modes to selected inference policy", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /fetch\(apiBase \+ "\/health"\)/);
  assert.match(page, /data\.modes/);
  assert.match(page, /nextModes\[mode\.key\]\?\.ready/);
  assert.match(page, /body\.append\("models", selected\.join\(","\)\)/);
  assert.match(page, /body\.append\("decision_mode", activeMode\.strategy\)/);
  assert.match(page, /fetch\(apiBase \+ "\/infer", \{ method: "POST", body \}\)/);
  assert.match(page, /availableCount !== selected\.length/);
  assert.match(page, /setDecision\(data\.decision/);
});
