"""Manual interactive agent enabling keyboard control over the paddle."""

import argparse
import asyncio
import contextlib
import json
import select
import sys
from typing import Any

import websockets

try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]
    HAS_TERMIOS = False


def _safe_print(text: str) -> None:
    """Print text, replacing characters the output encoding cannot represent (e.g. emoji on a cp1252 pipe)."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


async def receive_loop(
    websocket: Any, stop_event: asyncio.Event | None = None
) -> None:
    """Listen to server updates and print status to terminal."""
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "setup":
                print(f"\n[Handshake] Assigned Arkanoid Player ID: {data.get('player_id')}")
                print("Controls: 'A'/Left to move WEST, 'D'/Right to move EAST, 'Q' to quit.")
                print("=" * 60)
            elif msg_type in ("state", "update"):
                score = data.get("score", 0)
                lives = data.get("lives", 0)
                high_score = data.get("high_score", 0)
                timer = float(data.get("timer", data.get("time_elapsed", 0.0)))
                norm_score = float(data.get("normalized_score", 0.0))
                high_score_norm = float(data.get("high_score_normalized", 0.0))
                stage = data.get("stage", data.get("level", 1))
                max_stages = data.get("max_stages", data.get("max_levels", 3))
                paddle_x = data.get("paddle_x", 0.0)
                ball_x = data.get("ball_x", 0.0)
                ball_y = data.get("ball_y", 0.0)
                game_over = data.get("game_over", False)
                game_won = data.get("game_won", False)
                bricks = data.get("bricks", [])

                level_limit = float(data.get("level_time_limit", 0.0))
                level_elapsed = float(data.get("level_time_elapsed", 0.0))
                if level_limit > 0.0:
                    time_info = (
                        f"TIME LEFT: {max(0.0, level_limit - level_elapsed):05.1f}s"
                    )
                else:
                    time_info = f"TIME: {timer:05.1f}s"

                # Clean screen redraw
                sys.stdout.write("\033[H\033[2J")
                sys.stdout.flush()

                print("=" * 20 + " ARKANOID HUD " + "=" * 20)
                print(
                    f"STAGE: {stage}/{max_stages} | {time_info} | LIVES: {lives}"
                )
                print(
                    f"SCORE: {score:<5} ({norm_score:.1f}/s) | HIGH: {high_score:<5} ({high_score_norm:.1f}/s)"
                )
                print("-" * 54)
                print(
                    f"Paddle X: {paddle_x:<8.1f} | Ball: ({ball_x:.1f}, {ball_y:.1f})"
                )
                print(f"Active Bricks: {len(bricks)}")

                if game_won:
                    _safe_print("=" * 16 + " 🏆 YOU WON! ALL STAGES CLEARED! 🏆 " + "=" * 16)
                    print("\nGame session completed. Stopping agent process.")
                    if stop_event is not None:
                        stop_event.set()
                    break
                if game_over:
                    _safe_print("=" * 20 + " 💥 GAME OVER! 💥 " + "=" * 20)
                    print("\nGame session completed. Stopping agent process.")
                    if stop_event is not None:
                        stop_event.set()
                    break

                print("=" * 54)
                print("\n[CONTROLS] A / Left: West | D / Right: East | Q: Quit")

    except websockets.ConnectionClosed:
        print("\nDisconnected from Arkanoid Server.")
    finally:
        if stop_event is not None:
            stop_event.set()


async def send_loop(
    websocket: Any, stop_event: asyncio.Event | None = None
) -> None:
    """Capture raw terminal keystrokes and transmit move actions."""
    is_tty = HAS_TERMIOS and sys.stdin.isatty()
    fd = sys.stdin.fileno() if is_tty else None
    old_settings = None

    if is_tty and fd is not None and termios is not None and tty is not None:
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    try:
        while stop_event is None or not stop_event.is_set():
            key = ""
            if is_tty and fd is not None:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    # Handle escape sequences for arrow keys
                    if ch == "\x1b":
                        seq = sys.stdin.read(2)
                        if seq == "[D":
                            key = "a"  # Left Arrow
                        elif seq == "[C":
                            key = "d"  # Right Arrow
                    else:
                        key = ch
            else:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    line = sys.stdin.readline().strip().lower()
                    if line in ("a", "d", "q"):
                        key = line

            if key:
                key_lower = key.lower()
                if key_lower == "q":
                    if stop_event is not None:
                        stop_event.set()
                    break
                if key_lower in ("a", "west"):
                    await websocket.send(
                        json.dumps({"action": "move", "direction": "WEST"})
                    )
                elif key_lower in ("d", "east"):
                    await websocket.send(
                        json.dumps({"action": "move", "direction": "EAST"})
                    )

            await asyncio.sleep(0.02)
    finally:
        if (
            is_tty
            and fd is not None
            and old_settings is not None
            and termios is not None
        ):
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting Manual Agent...")


async def run_manual_agent(
    url: str = "ws://localhost:8765/ws",
    name: str = "Manual Arkanoid Driver",
) -> None:
    """Connect to WebSocket server and launch send/receive tasks."""
    print(f"Connecting to Arkanoid Server on {url}...")
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(
                json.dumps({"client": "agent", "name": name})
            )
            stop_event = asyncio.Event()
            rec_task = asyncio.create_task(receive_loop(websocket, stop_event))
            snd_task = asyncio.create_task(send_loop(websocket, stop_event))
            await asyncio.wait(
                [rec_task, snd_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop_event.set()
            for task in (rec_task, snd_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
    except Exception as e:
        print(f"Connection error: {e}")


def main() -> None:
    """CLI entry point for manual agent."""
    parser = argparse.ArgumentParser(description="Arkanoid manual agent")
    parser.add_argument("-s", "--server", default="ws://localhost:8765/ws", help="WebSocket URI")
    parser.add_argument(
        "-n", "--name", default="Manual Arkanoid Driver", help="Player name shown in the high score table"
    )
    args = parser.parse_args()
    if not args.name.strip():
        parser.error("--name must not be empty")
    try:
        asyncio.run(run_manual_agent(args.server, args.name.strip()))
    except KeyboardInterrupt:
        print("\nManual agent stopped.")


if __name__ == "__main__":
    main()
