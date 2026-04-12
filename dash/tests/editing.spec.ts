import { test, expect } from '@playwright/test';

test('ctrl+w and ctrl+u', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await expect(page.locator('.xterm')).toBeVisible();
  await expect(page.locator('.xterm-rows')).toContainText('Shell loaded');
  await page.waitForTimeout(500);

  await page.locator('.xterm').click();
  // Type a wrong command, hit Ctrl+U, then type uname
  await page.keyboard.type('wrong command', { delay: 50 });
  await page.keyboard.press('Control+U');
  await page.keyboard.type('uname\r', { delay: 50 });

  await expect(page.locator('.xterm-rows')).toContainText('Emscripten');

  // Type unxzz zz, hit Ctrl+W (erases zz), hit Ctrl+W (erases unxzz), type ls /
  await page.keyboard.type('unxzz zz', { delay: 50 });
  await page.keyboard.press('Control+W');
  await page.keyboard.press('Control+W');
  await page.keyboard.type('ls /\r', { delay: 50 });

  await expect(page.locator('.xterm-rows')).toContainText('bin');
});
