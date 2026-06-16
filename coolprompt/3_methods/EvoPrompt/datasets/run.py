"""Entry point for running EvoPrompt with ``gpt-5-nano`` over a generic dataset.

Example
-------

::

    export OPENAI_API_KEY=sk-...
    python run.py --dataset gsm8k --evo_mode de --popsize 10 --budget 10 \
                  --sample_num 50 --test_sample_num 100 \
                  --output outputs/gsm8k_de --seed 5
"""

from __future__ import annotations

import os
import sys

from utils import set_seed
from args import parse_args
from llm_client import llm_init


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[run.py] ERROR: OPENAI_API_KEY is not set and --openai_api_key "
              "was not provided.", file=sys.stderr)
        sys.exit(1)

    llm_config = llm_init(
        model=args.model,
        temperature=args.temperature,
        api_key="api_key",
        base_url=args.openai_base_url,
    )

    # ``client`` is kept for signature compatibility with the original code; the
    # singleton ChatOpenAI is stored inside ``llm_client``.
    client = None

    if args.evo_mode == "de":
        from evoluter import DEEvoluter
        evoluter = DEEvoluter(args, llm_config, client)
    elif args.evo_mode == "ga":
        from evoluter import GAEvoluter
        evoluter = GAEvoluter(args, llm_config, client)
    elif args.evo_mode == "ape":
        from evoluter import ParaEvoluter
        evoluter = ParaEvoluter(args, llm_config, client)
    else:
        raise ValueError(f"Unknown --evo_mode {args.evo_mode}")

    evoluter.evolute()

    print(f"\n[run.py] Optimization log written to: {args.results_json}")
    final = evoluter.history.get("final", {})
    if final:
        print(f"[run.py] dataset={final.get('dataset')} "
              f"metric={final.get('metric')}")
        print(f"[run.py] Best prompt: {final.get('best_prompt')!r}")
        print(f"[run.py] dev_score={final.get('dev_score')} "
              f"test_score={final.get('test_score')}")


if __name__ == "__main__":
    main()
