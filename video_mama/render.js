// Рендер кадров через Chromium (Playwright) и склейка в MP4 через ffmpeg.
// node render.js out.mp4            — весь ролик (без звука)
// node render.js --stills 0,25,40   — отдельные кадры в stills/*.png
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
let pw;
try { pw = require('playwright'); } catch { pw = require('/opt/node22/lib/node_modules/playwright'); }

const FPS = 30;
const SCALE = 1080 / 576; // 576x1024 -> 1080x1920

(async () => {
  const args = process.argv.slice(2);
  const browser = await pw.chromium.launch();
  const page = await browser.newPage({ viewport: { width: 576, height: 1024 }, deviceScaleFactor: SCALE });
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
    await browser.close();
    return;
  }

  if (args[0] === '--timeline') {
    // тайминги для звука
    const data = await page.evaluate(() => ({ keys: window.KEYS, msgs: window.MSGS, drop: window.SCENE.drop, duration: window.SCENE.duration }));
    fs.writeFileSync(path.join(__dirname, 'timeline.json'), JSON.stringify(data, null, 1));
    await browser.close();
    return;
  }

  const out = args[0] || 'video_noaudio.mp4';
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const frames = Math.round(duration * FPS);
  const t0 = Date.now();
  for (let f = 0; f < frames; f++) {
    await page.evaluate(t => window.render(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 150 === 0) console.log(`кадр ${f}/${frames}  ${((Date.now() - t0) / 1000).toFixed(0)}с`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.log('готово:', out);
})();
