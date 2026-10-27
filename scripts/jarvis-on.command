#!/bin/bash
# Включить Джарвиса. Двойной клик в Finder.
PROJECT="/Users/jigo/Documents/PETs/projects/jarvis-agent"
cd "$PROJECT" || exit 1
echo "== Включить Джарвиса"
/usr/bin/python3 -m src.jarvis_backend.autostart on
echo
read -n 1 -s -r -p "Готово. Нажмите любую клавишу, чтобы закрыть окно."
