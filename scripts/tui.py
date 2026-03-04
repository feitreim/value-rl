import subprocess
import os
import sys

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Tome TUI Wrapper")
        print("\nUsage: uv run tui [options]")
        print("\nNote: This is a Python wrapper around 'cargo run --release' in Tome/tui.")
        return

    tui_dir = os.path.join(os.getcwd(), "Tome", "tui")
    print(f"=== Tome TUI Startup ===")
    
    # Build if needed
    if not os.path.exists(os.path.join(tui_dir, "target", "release", "tui")):
        print("Building TUI (release)...")
        subprocess.run(["cargo", "build", "--release"], cwd=tui_dir, check=True)
    
    try:
        # PAGER=cat to avoid terminal pagination
        env = os.environ.copy()
        env["PAGER"] = "cat"
        cmd = ["cargo", "run", "--release"]
        if len(sys.argv) > 1:
            cmd.append("--")
            cmd.extend(sys.argv[1:])
            
        subprocess.run(cmd, cwd=tui_dir, env=env, check=True)
    except KeyboardInterrupt:
        print("\nStopping TUI...")
    except Exception as e:
        print(f"Error starting TUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
