import { expect, test, type Page } from "@playwright/test";

const papers = [0, 1, 2].map(i => ({
  i, id: `2401.0000${i}`, t: `Research paper ${i}`, au: "A Researcher", d: "2024-01-02",
  c: i === 2 ? -1 : 0, cat: "cs.AI", x: i, y: i, ci: i, rf: 0, v: "",
  ab: "A readable abstract about machine learning and research discovery.",
}));
const data = {
  version: "test", papers, edges: [[0, 1, .8]], neighbors: { "0": [[1, .8]], "1": [[0, .8]] },
  areas: [{ id: 0, name: "Machine learning", color: "#2f6feb", size: 2 }],
  categories: [["cs.AI", 3]], sizeBy: "links",
};

async function render(page: Page, overrides = {}) {
  await page.evaluate(args => window.postMessage({ type: "streamlit:render", args }, "*"), {
    data, highlight: null, focus_area: null, select: null, height: 960, saved_ids: [], ...overrides,
  });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toHaveText("Loading your research workspace…");
});

test("area to paper, stable hover, related work and keyboard selection", async ({ page }) => {
  await render(page);
  await page.getByRole("button", { name: /Explore papers/ }).click();
  await expect(page.locator(".row")).toHaveCount(2);
  await page.locator(".row").first().focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await page.locator(".row").nth(1).hover();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await page.locator(".sim-row").click();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 1");
});

test("empty search never falls back to unrelated papers", async ({ page }) => {
  await render(page, { highlight: { query: "missing topic", items: [], req: 1 } });
  await expect(page.getByText("No papers to show.", { exact: false })).toBeVisible();
  await expect(page.locator(".row")).toHaveCount(0);
});

test("index zero preselection and mobile back navigation keep details accessible", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await render(page, { select: 0 });
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await expect(page.getByRole("link", { name: "Open on arXiv" })).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Back to papers", exact: false }).click();
  await expect(page.locator(".row").first()).toBeVisible();
  await page.locator(".row").nth(1).click();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 1");
});

test("save event and updated saved state preserve selection", async ({ page }) => {
  await page.evaluate(() => {
    (window as any).outgoing = [];
    window.addEventListener("message", e => {
      if (e.data.type === "streamlit:setComponentValue") (window as any).outgoing.push(e.data.value);
    });
  });
  await render(page, { select: 0 });
  await page.getByRole("button", { name: "Save to reading list" }).click();
  await expect.poll(() => page.evaluate(() => (window as any).outgoing.at(-1)?.id)).toBe("2401.00000");
  await render(page, { saved_ids: ["2401.00000"] });
  await expect(page.getByRole("button", { name: "Remove from reading list" })).toBeVisible();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
});

test("search results replace a previous area's filter and preserve ranked order", async ({ page }) => {
  await render(page, { focus_area: 0 });
  await render(page, { highlight: { query: "related topic", items: [[2, .9], [0, .8]], req: 2 } });
  await expect(page.locator(".row-title")).toHaveText(["Research paper 2", "Research paper 0"]);
  await expect(page.getByLabel("Filter by research area")).toHaveValue("");
});
