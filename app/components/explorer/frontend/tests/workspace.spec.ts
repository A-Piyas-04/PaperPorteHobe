import { expect, test, type Page } from "@playwright/test";

const papers = [0, 1, 2].map(i => ({
  i, id: `2401.0000${i}`, t: `Research paper ${i}`, au: "A Researcher", d: "2024-01-02",
  c: i === 2 ? -1 : 0, cat: "cs.AI", x: i, y: i, ci: i, rf: 0, v: "",
  ab: "A readable abstract about machine learning and research discovery.",
}));
const data = {
  version: "test", papers, edges: [[0, 1, .8]], neighbors: { "0": [[1, .8]], "1": [[0, .8]] },
  areas: [{ id: 0, name: "Machine learning", color: "#2f6feb", size: 2, rc: 2, x: 0.5, y: 0.5 }],
  areaEdges: [], categories: [["cs.AI", 3]], sizeBy: "links",
  corpusTotal: 3, renderedTotal: 3, maxAreaPapers: 400,
};

async function render(page: Page, overrides = {}) {
  await page.evaluate(args => window.postMessage({ type: "streamlit:render", args }, "*"), {
    data, highlight: null, focus_area: null, select: null, height: 960, saved_ids: [], ...overrides,
  });
}

const openList = (page: Page) => page.getByRole("button", { name: /List/ }).click();

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toHaveText("Loading your research workspace…");
});

test("the connected map is the default view and explains how to read it", async ({ page }) => {
  await render(page);
  await expect(page.locator(".map-view")).toBeVisible();
  await expect(page.getByRole("button", { name: "🗺️ Map" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".crumb.current")).toHaveText("Overview");
  await expect(page.getByText("How to read this map")).toBeVisible();
  // Node colour, size and edge meaning are all disclosed; similarity is not a score.
  await expect(page.getByText(/not a citation or a confidence score/)).toBeVisible();
});

test("list view: keyboard selection, stable hover and related work", async ({ page }) => {
  await render(page);
  await openList(page);
  await page.getByRole("button", { name: /Explore papers/ }).click();
  await expect(page.locator(".row")).toHaveCount(2);
  await page.locator(".row").first().focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  // Hover must never change the selection or the details.
  await page.locator(".row").nth(1).hover();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await page.locator(".sim-row").click();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 1");
});

test("empty search falls back to unrelated papers on neither map nor list", async ({ page }) => {
  await render(page, { highlight: { query: "missing topic", items: [], req: 1 } });
  await expect(page.getByText("No papers matched this search", { exact: false })).toBeVisible();
  await openList(page);
  await expect(page.getByText("No papers to show.", { exact: false })).toBeVisible();
  await expect(page.locator(".row")).toHaveCount(0);
});

test("paper index zero preselects, and mobile details open and return to the map", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await render(page, { select: 0 });
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await expect(page.getByRole("link", { name: "Open on arXiv" })).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Back to map", exact: false }).click();
  await expect(page.locator(".map-drawer")).toBeHidden();
});

test("saving a paper keeps the selection (and does not reset the map)", async ({ page }) => {
  await page.evaluate(() => {
    (window as any).outgoing = [];
    window.addEventListener("message", e => {
      if (e.data.type === "streamlit:setComponentValue") (window as any).outgoing.push(e.data.value);
    });
  });
  await render(page, { select: 0 });
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await page.getByRole("button", { name: "Save to reading list" }).click();
  await expect.poll(() => page.evaluate(() => (window as any).outgoing.at(-1)?.id)).toBe("2401.00000");
  await render(page, { select: 0, saved_ids: ["2401.00000"] });
  await expect(page.getByRole("button", { name: "Remove from reading list" })).toBeVisible();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
});

test("search results replace an area focus and keep ranked order", async ({ page }) => {
  await render(page, { focus_area: 0 });
  await render(page, { highlight: { query: "related topic", items: [[2, .9], [0, .8]], req: 2 } });
  await openList(page);
  await expect(page.locator(".row-title")).toHaveText(["Research paper 2", "Research paper 0"]);
  await expect(page.getByLabel("Filter by research area")).toHaveValue("");
});

test("selection is shared when switching between list and map", async ({ page }) => {
  await render(page);
  await openList(page);
  await page.getByRole("button", { name: /Explore papers/ }).click();
  await page.locator(".row").first().click();
  await expect(page.locator(".detail-title")).toHaveText("Research paper 0");
  await page.getByRole("button", { name: "🗺️ Map" }).click();
  // The map drawer shows the same selected paper.
  await expect(page.locator(".map-drawer .detail-title")).toHaveText("Research paper 0");
});
