import argparse
import requests
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Stream from local Ollama")
    parser.add_argument("prompt", nargs="+", help="Your prompt")
    parser.add_argument("-m", "--model", default="llama3.2", help="model name")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="temperature")
    parser.add_argument("-p", "--port", type=int, default=11434, help="Ollama port (default: 11434)")
    parser.add_argument("--host", default="localhost", help="host (default: localhost)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/api/generate"

    payload = {
        "model": args.model,
        "prompt": " ".join(args.prompt),
        "stream": True,
        "options": {"temperature": args.temp}
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            print(data["response"], end="", flush=True)
                        if data.get("done", False):
                            print()
                            break
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()