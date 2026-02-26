"""
Measure tokens/second with 3 concurrent "users" talking to Ollama
Model: qwen2:0.5b or qwen:0.5b or qwen2.5:0.5b (depending on your tag)

Requires: ollama running locally
"""

import ollama
import random
import time
import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class Result:
    name: str
    tokens_per_sec: float
    total_tokens: int
    duration_sec: float
    error: Optional[str] = None

def run_one_user(name: str, prompt: str, model: str = "qwen2:0.5b") -> Result:
    try:
        start_time = time.time()
        total_tokens = 0

        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options={
                "temperature": 0.7,
                # "num_predict": 400,     # uncomment if you want to limit output length
            }
        )

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                # Rough token estimation (not perfect but good enough for speed comparison)
                total_tokens += max(1, len(content.strip()) // 4 + 1)

            if 'eval_count' in chunk.get('eval', {}):
                # Newer ollama sometimes puts final stats here
                total_tokens = chunk['eval']['eval_count']
                break

        end_time = time.time()
        duration = end_time - start_time

        if total_tokens <= 0:
            return Result(name, 0.0, 0, duration, "No tokens counted")

        tps = total_tokens / duration

        return Result(name, tps, total_tokens, duration)

    except Exception as e:
        return Result(name, 0.0, 0, 0, str(e))


def worker(name: str, prompt: str, results: list, model: str):
    res = run_one_user(name, prompt, model)
    results.append(res)


def main():
    # MODEL = "qwen2:0.5b"          # change to "qwen:0.5b" or "qwen2.5:0.5b" if needed
    MODEL = "gemma3:12b"        # you can also try slightly bigger model

    prompts = [
        "Write me a savage roast for my friend who always cancels plans",
        "Explain quantum entanglement like I'm 12 years old",
        "Give me 8 passive-aggressive ways to tell someone they're late",
        "Turn this terrible Tinder bio into a god-tier version: 'I like food and breathing'",
        "What's the most unhinged workout plan that still kinda works?",
        "Write a break-up text that's cold but not cruel",
        "Rank these 5 side hustles from most to least likely to make me $5k/month in Vietnam",
        "Describe how my cat sees me in the most dramatic way possible",
        "Create a 1-week meal plan under 2 million VND for someone who hates cooking",
        "Give me 10 unhinged Reddit AITA-style titles that are actually real",
        "Write a motivational speech from the perspective of my future rich self",
        "How would Sun Tzu win a modern office politics war?",
        "Turn my boring CV bullet points into LinkedIn brag versions",
        "Explain why Gen Z hates phone calls but loves voice messages",
        "Write angry Vietnamese auntie energy text to a scammer",
        "Give me 7 savage comebacks for when someone says 'calm down'",
        "Create the most cursed dating app opener possible",
        "What's the Vietnamese equivalent of 'main character energy'?",
        "Write a resignation letter that burns every bridge elegantly",
        "Explain crypto crashes using only food metaphors",
        "Make a sleep paralysis demon argue why I should stay awake",
        "Give me a 3-day Ho Chi Minh City itinerary for broke university students",
        "Write the most dramatic apology text after ghosting someone",
        "Rank programming languages by how much they make developers suffer",
        "Describe my personality based only on my Spotify Wrapped top 5",
        "Create 5 Instagram captions for when I look good but feel dead inside",
        "How to survive family Tet holiday questions about marriage and salary",
        "Write a love letter from a keyboard to a gamer who never cleans it",
        "Explain inflation in Vietnam like I'm a goldfish with 3-second memory",
        "Give me the ultimate petty revenge plan that stays legal"
    ]
    concurrent_request_numbers = 1
    random.seed(42)
    print(f"\nStarting {concurrent_request_numbers} concurrent requests to ollama ({MODEL})\n")
    
    random_requests = random.sample(prompts, concurrent_request_numbers)

    results = []
    threads = []

    start_all = time.time()

    for i, prompt in enumerate(random_requests, 1):
        t = threading.Thread(
            target=worker,
            args=(f"user-{i}", prompt, results, MODEL)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_all = time.time()
    total_duration = end_all - start_all

    print("\n" + "="*70)
    print(f"Results (model: {MODEL})")
    print("-"*70)

    for r in sorted(results, key=lambda x: x.name):
        if r.error:
            print(f"{r.name:8} → ERROR: {r.error}")
        else:
            print(f"{r.name:8} → {r.tokens_per_sec:6.1f} t/s   "
                  f"({r.total_tokens:3d} tokens in {r.duration_sec:5.2f}s)")

    print("-"*70)
    print(f"Wall-clock time for all 3 requests: {total_duration:.2f} seconds")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()