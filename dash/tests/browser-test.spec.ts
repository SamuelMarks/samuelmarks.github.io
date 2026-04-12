import { test, expect } from '@playwright/test';

test('dash terminal boots and runs commands', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await expect(page.locator('.xterm')).toBeVisible();
  await expect(page.locator('.xterm-rows')).toContainText('Shell loaded');
  await page.waitForTimeout(500);

  // Test uname
  await page.locator('.xterm').click();
  await page.keyboard.type('uname\r', { delay: 50 });
  await expect(page.locator('.xterm-rows')).toContainText('Emscripten');

  await page.waitForTimeout(500);

  // Test ls
  await page.keyboard.type('ls /\r', { delay: 50 });
  await expect(page.locator('.xterm-rows')).toContainText('bin');
  await expect(page.locator('.xterm-rows')).toContainText('tmp');
  await expect(page.locator('.xterm-rows')).toContainText('home');

  await page.waitForTimeout(500);

  // Test cat
  await page.keyboard.type('cat /etc/profile\r', { delay: 50 });
  await expect(page.locator('.xterm-rows')).toContainText('export PATH=/bin');
});
