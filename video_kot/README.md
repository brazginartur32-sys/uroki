# «Чья будете» — короткое квадратное видео в стиле переписки Telegram

Готовый ролик: `kot.mp4` (1080×1080, 30 fps, 20,9 с, звук).

Монтаж повторяет референс покадрово: наплыв из чёрного, допечатка и отправка, подсказка
«Удерживайте для записи звука», запись голосового (таймер ×2), закрепление, входящие во время записи,
отмена в корзину, ответ с опечаткой и автозаменой, рывки-зумы со смазом на каждом сообщении,
медленный наезд камеры, глитч и обрыв в чёрное. История, стикер (кошечка с ромашками), музыка — свои.

## Как поменять текст и пересобрать
Тексты и тайминг — в `scene.js`. Сборка:
```bash
node render.js --timeline                      # тайминги для звука
python3 audio.py timeline.json audio.wav       # музыка и звуки (нужен numpy)
node render.js frames                          # кадры (Playwright + Chromium)
python3 post.py frames timeline.json video_noaudio.mp4   # наплыв, наезд, глитч (numpy + Pillow)
ffmpeg -y -i video_noaudio.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart kot.mp4
```
Отдельные кадры для проверки: `node render.js --stills 1,5.6,19.7`.

Шрифт Roboto (SIL Open Font License). Стикер нарисован с нуля (`sticker.svg`).
