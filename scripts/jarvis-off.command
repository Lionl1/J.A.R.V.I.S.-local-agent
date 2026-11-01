#!/bin/bash
# ВЫКЛЮЧИТЬ Джарвиса: микрофон, автозапуск, модель. Двойной клик в Finder.
PROJECT="/Users/jigo/Documents/PETs/projects/jarvis-agent"
cd "$PROJECT" || exit 1
echo "== ВЫКЛЮЧИТЬ Джарвиса: микрофон, автозапуск, модель"
/usr/bin/python3 -m src.jarvis_backend.autostart off
echo
read -n 1 -s -r -p "Готово. Нажмите любую клавишу, чтобы закрыть окно."
