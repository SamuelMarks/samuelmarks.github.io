import { test, expect } from '@playwright/test';

test('backspace editing', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await expect(page.locator('.xterm')).toBeVisible();
  await expect(page.locator('.xterm-rows')).toContainText('Shell loaded');
  await page.waitForTimeout(500);

  await page.locator('.xterm').click();
  // Type unzme, backspace twice, type ame -> uname
  await page.keyboard.type('unzme', { delay: 50 });
  await page.keyboard.press('Backspace');
  await page.keyboard.press('Backspace');
  await page.keyboard.press('Backspace');
  await page.keyboard.type('ame\r', { delay: 50 });

  await expect(page.locator('.xterm-rows')).toContainText('Emscripten');
});
