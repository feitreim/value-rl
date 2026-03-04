import subprocess
import os
import sys

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Tome Scheduler Wrapper")
        print("\nUsage: uv run scheduler [options]")
        print("\nNote: This is a Python wrapper around 'cargo run --release' in Tome/scheduler.")
        print("Set SCHEDULER_PORT environment variable to change the port (default: 8080).")
        return

    scheduler_dir = os.path.join(os.getcwd(), "Tome", "scheduler")
    port = os.environ.get("SCHEDULER_PORT", "8080")
    print(f"=== Tome Scheduler Startup ===")
    print(f"Starting Tome Scheduler on port {port}...")
    
    # Build if needed
    if not os.path.exists(os.path.join(scheduler_dir, "target", "release", "scheduler")):
        print("Building scheduler (release)...")
        subprocess.run(["cargo", "build", "--release"], cwd=scheduler_dir, check=True)
    
    try:
        # PAGER=cat to avoid terminal pagination
        env = os.environ.copy()
        env["PAGER"] = "cat"
        cmd = ["cargo", "run", "--release"]
        if len(sys.argv) > 1:
            cmd.append("--")
            cmd.extend(sys.argv[1:])
            
        subprocess.run(cmd, cwd=scheduler_dir, env=env, check=True)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
    except Exception as e:
        print(f"Error starting scheduler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
