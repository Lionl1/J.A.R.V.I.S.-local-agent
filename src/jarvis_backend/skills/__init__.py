"""Навыки — инструменты, которые Джарвис может дописывать себе сам.

Каждый модуль в этом пакете (кроме начинающихся с "_") — один навык:

    TOOLS = [                       # схемы в формате OpenAI function calling
        {"type": "function", "function": {"name": "my_tool", "description": "...",
                                          "parameters": {"type": "object", "properties": {...}}}},
    ]
    CONFIRM = {"my_tool"}           # необязательно: инструменты, требующие «Разрешить»

    async def my_tool(arg: str) -> dict | str:   # sync-функции тоже допустимы
        ...

Имя функции = имя инструмента. Навыки перезагружаются сразу после правки
(update_source_code / edit_source_code) — перезапуск сервера не нужен.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class SkillSet:
    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    schemas: list[dict[str, Any]] = field(default_factory=list)
    confirm: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)


def load_skills(reserved_names: set[str] | frozenset[str] = frozenset()) -> SkillSet:
    """(Пере)импортирует все навыки. Сломанный навык не мешает остальным."""
    result = SkillSet()
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        full_name = f"{__name__}.{info.name}"
        try:
            if full_name in sys.modules:
                module = importlib.reload(sys.modules[full_name])
            else:
                module = importlib.import_module(full_name)
            tools = getattr(module, "TOOLS", [])
            confirm = set(getattr(module, "CONFIRM", set()))
            for schema in tools:
                name = schema["function"]["name"]
                func = getattr(module, name, None)
                if not callable(func):
                    raise ValueError(f"в TOOLS есть '{name}', но нет функции с таким именем")
                if name in reserved_names or name in result.functions:
                    raise ValueError(f"имя инструмента '{name}' уже занято")
                result.functions[name] = func
                result.schemas.append(schema)
                if name in confirm:
                    result.confirm.add(name)
        except Exception as exc:  # noqa: BLE001
            result.errors[info.name] = f"{type(exc).__name__}: {exc}"
    return result
