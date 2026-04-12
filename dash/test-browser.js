const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  await page.goto('http://localhost:4174');
  await page.waitForTimeout(1000); // let dash load
  
  await page.locator('.xterm-helper-textarea').focus();
  await page.keyboard.type('ls -l');
  await page.keyboard.press('Enter');
  
  await page.waitForTimeout(2000); // wait for output
  
  const content = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('.xterm-rows > div')).map(row => row.textContent).join('\n');
  });
  console.log("--- TERMINAL OUTPUT ---");
  console.log(content);
  
  await browser.close();
  process.exit(0);
})();
