# «Просто потому, что это весело» — мрачная переписка в Telegram

Готовый ролик: `park.mp4` (1080×1080, 30 fps, 32,4 с, звук). Монтаж как в `video_kot`
(рывки со смазом, «перезалив», 20 fps), плюс помехи телевизора «Спустя час…»,
затухание экрана с каждым сообщением (темнеет фон и края, старые сообщения уходят в тень, новые читаемы),
глитч в стиле референса без скачков яркости, наезд и резкая темнота.

Тексты и тайминг — `scene.js`. Сборка:
```bash
node render.js --timeline
python3 audio.py timeline.json audio.wav
node render.js frames
python3 post.py frames timeline.json video_noaudio.mp4
ffmpeg -y -i video_noaudio.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart park.mp4
```

## Проверка безопасности и читаемости
```bash
node render.js --layout
python3 check_watch.py park.mp4 layout.json timeline.json
```
Считает вспышки по WCAG 2.3.1 / методике Harding (опасно — больше 3 в секунду), красные вспышки
и контраст текста каждого сообщения по кадрам (норма WCAG — от 4.5:1).
Текущая версия: не больше 1 вспышки в секунду, контраст сообщений от 8.3:1.
