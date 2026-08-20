// Render the viewer headless and save reference screenshots for review.
import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';

const here = path.dirname(new URL(import.meta.url).pathname);
const viewer = 'file://' + path.join(here, 'out', process.env.VIEWER || 'viewer.html');
const outDir = process.argv[3] || path.join(here, 'out', 'shots');
fs.mkdirSync(outDir, { recursive: true });

const shots = (process.argv[2] || 'photo,front,angle,corner,handle,laths,meeting,box,outside')
  .split(',').filter(Boolean);

const browser = await chromium.launch({
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
         '--disable-lcd-text', '--force-device-scale-factor=1'],
});
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
page.on('console', m => { if (m.type() === 'error') console.log('  [console]', m.text()); });

for (const s of shots) {
  const [name, extra] = s.split('|');
  const url = `${viewer}?view=${name}&ss=1.5${extra ? '&' + extra : ''}`;
  await page.goto(url);
  await page.waitForFunction('window.__ready === true', { timeout: 60000 });
  await page.waitForTimeout(450);
  const err = await page.$eval('#err', e => e.textContent);
  if (err) console.log(`  !! ${name}: ${err}`);
  const file = path.join(outDir, `${name}.png`);
  await page.screenshot({ path: file });
  console.log('  saved', path.relative(here, file));
}
await browser.close();
