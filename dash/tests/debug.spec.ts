import { test, expect } from '@playwright/test';

test('debug manual typing', async ({ page }) => {
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  await page.goto('http://localhost:5173/');
  await expect(page.locator('.xterm')).toBeVisible();
  await expect(page.locator('.xterm-rows')).toContainText('Shell loaded');
  await page.waitForTimeout(500);

  await page.locator('.xterm').click();
  await page.keyboard.type('uname\r', { delay: 50 });

  await page.waitForTimeout(2000);
  
  const text = await page.locator('.xterm-rows').innerText();
  console.log("XTERM TEXT:\n", text);
});
