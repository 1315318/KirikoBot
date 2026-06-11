from __future__ import annotations

from typing import Any


class MsgPackage:
    """Assembles OneBot message segments from a type template."""

    def robot_server_msg(
        self, msg_type_dict: dict[str, str], robot_server: Any
    ) -> None:
        msg_type = msg_type_dict.get("type", "")
        robot_server.msg_list: list[dict[str, Any]] = []

        if "image" in msg_type:
            robot_server.msg_list.append(
                {"type": "image", "data": {"file": robot_server.image_path}}
            )
        if "at" in msg_type:
            robot_server.msg_list.append(
                {"type": "at", "data": {"qq": robot_server.user_id}}
            )
        if "text" in msg_type:
            robot_server.msg_list.append(
                {"type": "text", "data": {"text": robot_server.text}}
            )
