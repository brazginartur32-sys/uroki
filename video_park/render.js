// Рендер кадров через Chromium (Playwright).
// node render.js frames            — все кадры в frames/0000.jpg (дальше post.py)
// node render.js --stills 1,5.6    — отдельные кадры в stills/
// node render.js --timeline        — тайминги для звука в timeline.json
const path = require('path');
const fs = require('fs');
let pw;
try { pw = require('playwright'); } catch { pw = require('/opt/node22/lib/node_modules/playwright'); }

const FPS = 30;
const SCALE = 1080 / 576;

(async () => {
  const args = process.argv.slice(2);
  const browser = await pw.chromium.launch();
  const page = await browser.newPage({ viewport: { width: 576, height: 576 }, deviceScaleFactor: SCALE });
  await page.goto('file://' + path.join(__dirname, 'index.html'));
  await page.evaluate(() => window.ready);
  const duration = await page.evaluate(() => window.SCENE.duration);

  if (args[0] === '--stills') {
    const dir = path.join(__dirname, 'stills');
    fs.mkdirSync(dir, { recursive: true });
    for (const t of args[1].split(',').map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await page.screenshot({ path: path.join(dir, `t_${t.toFixed(2)}.png`) });
    }
  } else if (args[0] === '--timeline') {
    const data = await page.evaluate(() => ({ keys: window.KEYS, msgs: window.MSGS, t: window.SCENE.t, duration: window.SCENE.duration }));
    fs.writeFileSync(path.join(__dirname, 'timeline.json'), JSON.stringify(data, null, 1));
  } else {
    const dir = path.join(__dirname, args[0] || 'frames');
    fs.mkdirSync(dir, { recursive: true });
    const frames = Math.round(duration * FPS);
    for (let f = 0; f < frames; f++) {
      await page.evaluate(t => window.render(t), f / FPS);
      await page.screenshot({ path: path.join(dir, String(f).padStart(4, '0') + '.jpg'), type: 'jpeg', quality: 96 });
      if (f % 90 === 0) console.log(`кадр ${f}/${frames}`);
    }
  }
  await browser.close();
})();
