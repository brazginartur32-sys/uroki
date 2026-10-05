# «Самый главный страх каждой мамы...» — вертикальное видео в стиле переписки Telegram

Готовый ролик: `mama.mp4` (1080×1920, 30 fps, 71,7 с, звук).

## Как поменять текст и пересобрать
1. Правьте `scene.js`: заголовок, сообщения, время появления, что мама печатает.
2. Сборка:
   ```bash
   node render.js --timeline                 # тайминги для звука
   python3 audio.py timeline.json audio.wav  # музыка и звуки (нужен numpy)
   node render.js video_noaudio.mp4          # кадры (Playwright + Chromium)
   ffmpeg -y -i video_noaudio.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart mama.mp4
   ```
   Быстрый просмотр отдельных кадров: `node render.js --stills 1,30,72`.

Всё нарисовано и сгенерировано с нуля: интерфейс чата (HTML/CSS), обои с дудлами (SVG),
музыка и звуки (синтез в `audio.py`). Шрифт Inter (SIL Open Font License).
