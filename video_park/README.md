# «Просто потому, что это весело» — мрачная переписка в Telegram

Готовый ролик: `park.mp4` (1080×1080, 30 fps, 29,6 с, звук). Монтаж как в `video_kot`
(рывки со смазом, «перезалив», 20 fps), плюс помехи телевизора «Спустя час…»,
затухание экрана с каждым сообщением, блики, наезд и резкая темнота.

Тексты и тайминг — `scene.js`. Сборка:
```bash
node render.js --timeline
python3 audio.py timeline.json audio.wav
node render.js frames
python3 post.py frames timeline.json video_noaudio.mp4
ffmpeg -y -i video_noaudio.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart park.mp4
```
