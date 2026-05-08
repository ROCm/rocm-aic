"""Download and convert public chat datasets to the
LMCache conversation schema.

Supported datasets:
  - sharegpt:   ShareGPT52K (RyokoAI, ~90K convs)
  - lmsys:      LMSYS-Chat-1M (gated, HF token)
  - wildchat:   WildChat-1M  (gated, HF token)
  - longbench:  LongBench (THUDM, ~4.7K long-ctx)
  - vicuna:     ShareGPT Vicuna unfiltered (~53K)
  - ultrachat:  UltraChat (stingning, ~774K)
  - oasst:      OpenAssistant oasst1 (~10K trees)
"""
import html
import json
import re
import sys
from pathlib import Path


def _get_hf_download():
    """Lazy-import hf_hub_download."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except ImportError:
        print(
            "Error: the 'huggingface-hub' package is "
            "required.\n"
            "Install with: pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)


DATASET_CONFIGS = {
    "sharegpt": {
        "repo": "RyokoAI/ShareGPT52K",
        "files": [
            "sg_90k_part1.json",
            "sg_90k_part2.json",
        ],
        "type": "dataset",
    },
    "lmsys": {
        "repo": "lmsys/lmsys-chat-1m",
        "files": [
            "data/train-00000-of-00006.parquet",
        ],
        "type": "dataset",
        "note": "Gated dataset \u2014 requires HF token",
    },
    "wildchat": {
        "repo": "allenai/WildChat-1M",
        "files": [
            "data/train-00000-of-00014.parquet",
        ],
        "type": "dataset",
        "note": "Gated dataset \u2014 requires HF token",
    },
    "longbench": {
        "repo": "THUDM/LongBench",
        "files": ["data.zip"],
        "type": "dataset",
    },
    "vicuna": {
        "repo": "anon8231489123/"
                "ShareGPT_Vicuna_unfiltered",
        "files": [
            "ShareGPT_V3_unfiltered_cleaned"
            "_split.json",
        ],
        "type": "dataset",
    },
    "ultrachat": {
        "repo": "stingning/ultrachat",
        "files": ["train_0.jsonl"],
        "type": "dataset",
    },
    "oasst": {
        "repo": "OpenAssistant/oasst1",
        "files": [
            "data/train-00000-of-00001.parquet",
        ],
        "type": "dataset",
    },
}


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "prompter": "user",
}


def _normalize_role(raw_role: str) -> str:
    """Map dataset-specific role names to schema."""
    return ROLE_MAP.get(
        raw_role.lower().strip(), "user"
    )


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\x80-\x9f]"
)
_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"
    r"\u2060-\u2064\u2066-\u2069"
    r"\u202a-\u202e\ufeff\u00ad"
    r"\u034f\u180e]"
)
_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _sanitize_content(text: str) -> str:
    """Strip HTML tags and problematic unicode."""
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2028", "\n")
    text = text.replace("\u2029", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _PRIVATE_USE_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _convert_sharegpt_row(row):
    """Convert one ShareGPT JSON object."""
    conv_id = row.get("id", "")
    raw_turns = row.get("conversations", [])
    turns = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(
            turn.get("from", "user")
        )
        content = _sanitize_content(
            turn.get("value", "")
        )
        if content:
            turns.append(
                {"role": role, "content": content}
            )
    if len(turns) < 2:
        return None
    return {"id": str(conv_id), "turns": turns}


def _load_sharegpt(cfg, max_conversations):
    """Download and parse ShareGPT JSON files."""
    hf_hub_download = _get_hf_download()
    conversations = []
    for filename in cfg["files"]:
        if len(conversations) >= max_conversations:
            break

        print(f"  Downloading {filename}...")
        local = hf_hub_download(
            repo_id=cfg["repo"],
            filename=filename,
            repo_type="dataset",
        )

        print(f"  Parsing {filename}...")
        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            if len(conversations) >= max_conversations:
                break
            conv = _convert_sharegpt_row(row)
            if conv:
                conversations.append(conv)
            if (
                len(conversations) % 1000 == 0
                and len(conversations) > 0
            ):
                print(
                    f"  Converted "
                    f"{len(conversations)} "
                    f"conversations..."
                )

    return conversations


def _load_parquet_dataset(cfg, max_conversations,
                          converter_fn):
    """Download and parse a parquet-based dataset."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print(
            "Error: pandas is required for parquet "
            "datasets.\n"
            "Install with: pip install pandas "
            "pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)

    hf_hub_download = _get_hf_download()
    conversations = []
    for filename in cfg["files"]:
        if len(conversations) >= max_conversations:
            break

        print(f"  Downloading {filename}...")
        local = hf_hub_download(
            repo_id=cfg["repo"],
            filename=filename,
            repo_type="dataset",
        )

        print(f"  Parsing {filename}...")
        df = pd.read_parquet(local)
        for _, row in df.iterrows():
            if len(conversations) >= max_conversations:
                break
            conv = converter_fn(row)
            if conv:
                conversations.append(conv)
            if (
                len(conversations) % 1000 == 0
                and len(conversations) > 0
            ):
                print(
                    f"  Converted "
                    f"{len(conversations)} "
                    f"conversations..."
                )

    return conversations


def _convert_lmsys_row(row):
    """Convert one LMSYS-Chat-1M row."""
    conv_id = row.get(
        "conversation_id", ""
    )
    raw_turns = row.get("conversation", [])
    turns = []
    if isinstance(raw_turns, str):
        try:
            raw_turns = json.loads(raw_turns)
        except (json.JSONDecodeError, TypeError):
            return None
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(
            turn.get("role", "user")
        )
        content = _sanitize_content(
            turn.get("content", "")
        )
        if content:
            turns.append(
                {"role": role, "content": content}
            )
    if len(turns) < 2:
        return None
    return {"id": str(conv_id), "turns": turns}


def _convert_wildchat_row(row):
    """Convert one WildChat row."""
    conv_id = row.get(
        "conversation_hash", ""
    )
    raw_turns = row.get("conversation", [])
    turns = []
    if isinstance(raw_turns, str):
        try:
            raw_turns = json.loads(raw_turns)
        except (json.JSONDecodeError, TypeError):
            return None
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(
            turn.get("role", "user")
        )
        content = _sanitize_content(
            turn.get("content", "")
        )
        if content:
            turns.append(
                {"role": role, "content": content}
            )
    if len(turns) < 2:
        return None
    return {"id": str(conv_id), "turns": turns}


def _convert_longbench_row(row):
    """Convert one LongBench JSONL row to a 2-turn
    conversation (user=context+input, assistant=answer).
    """
    conv_id = row.get("_id", "")
    context = row.get("context", "")
    question = row.get("input", "")
    answers = row.get("answers", [])
    if not context and not question:
        return None
    if not answers:
        return None

    user_text = context
    if question:
        user_text = (
            f"{context}\n\n{question}"
            if context else question
        )
    user_content = _sanitize_content(user_text)
    answer_text = (
        answers[0] if isinstance(answers, list)
        else str(answers)
    )
    asst_content = _sanitize_content(answer_text)
    if not user_content or not asst_content:
        return None

    return {
        "id": str(conv_id),
        "turns": [
            {"role": "user",
             "content": user_content},
            {"role": "assistant",
             "content": asst_content},
        ],
    }


def _load_longbench(cfg, max_conversations):
    """Download LongBench data.zip and iterate the
    per-task JSONL files inside."""
    import tempfile
    import zipfile

    hf_hub_download = _get_hf_download()
    conversations = []

    print("  Downloading data.zip...")
    local_zip = hf_hub_download(
        repo_id=cfg["repo"],
        filename="data.zip",
        repo_type="dataset",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Extracting data.zip...")
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(tmpdir)

        data_dir = Path(tmpdir)
        jsonl_files = sorted(
            data_dir.rglob("*.jsonl")
        )
        if not jsonl_files:
            print(
                "  Warning: no .jsonl files found "
                "in data.zip",
                file=sys.stderr,
            )
            return conversations

        for jf in jsonl_files:
            if (
                len(conversations)
                >= max_conversations
            ):
                break
            print(f"  Parsing {jf.name}...")
            with open(
                jf, "r", encoding="utf-8"
            ) as f:
                for line in f:
                    if (
                        len(conversations)
                        >= max_conversations
                    ):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    conv = (
                        _convert_longbench_row(row)
                    )
                    if conv:
                        conversations.append(conv)
                    if (
                        len(conversations) % 1000
                        == 0
                        and len(conversations) > 0
                    ):
                        print(
                            f"  Converted "
                            f"{len(conversations)}"
                            f" conversations..."
                        )

    return conversations


def _convert_ultrachat_row(row):
    """Convert one UltraChat JSONL row.

    UltraChat rows have ``id`` and ``data`` where
    ``data`` is a list of alternating strings:
    [user, assistant, user, assistant, ...].
    """
    conv_id = row.get("id", "")
    data = row.get("data", [])
    if not isinstance(data, list) or len(data) < 2:
        return None

    turns = []
    for i, text in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        content = _sanitize_content(str(text))
        if content:
            turns.append(
                {"role": role, "content": content}
            )
    if len(turns) < 2:
        return None
    return {"id": str(conv_id), "turns": turns}


def _load_jsonl_dataset(cfg, max_conversations,
                        converter_fn):
    """Download and parse a JSONL-based dataset."""
    hf_hub_download = _get_hf_download()
    conversations = []
    for filename in cfg["files"]:
        if len(conversations) >= max_conversations:
            break

        print(f"  Downloading {filename}...")
        local = hf_hub_download(
            repo_id=cfg["repo"],
            filename=filename,
            repo_type="dataset",
        )

        print(f"  Parsing {filename}...")
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                if (
                    len(conversations)
                    >= max_conversations
                ):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conv = converter_fn(row)
                if conv:
                    conversations.append(conv)
                if (
                    len(conversations) % 1000 == 0
                    and len(conversations) > 0
                ):
                    print(
                        f"  Converted "
                        f"{len(conversations)} "
                        f"conversations..."
                    )

    return conversations


def _convert_oasst_dataset(cfg, max_conversations):
    """Download oasst1 parquet and reconstruct
    conversation trees from flat message rows.

    Extracts the longest root-to-leaf path from
    each English message tree.
    """
    try:
        import pandas as pd
    except ImportError:
        print(
            "Error: pandas is required for parquet "
            "datasets.\n"
            "Install with: pip install pandas "
            "pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)

    hf_hub_download = _get_hf_download()
    filename = cfg["files"][0]
    print(f"  Downloading {filename}...")
    local = hf_hub_download(
        repo_id=cfg["repo"],
        filename=filename,
        repo_type="dataset",
    )

    print(f"  Parsing {filename}...")
    df = pd.read_parquet(local)

    en_df = df[df["lang"] == "en"]

    children = {}
    msg_map = {}
    roots = []
    for _, row in en_df.iterrows():
        mid = row["message_id"]
        pid = row.get("parent_id", None)
        msg_map[mid] = row
        if pid is None or (
            isinstance(pid, float)
            and str(pid) == "nan"
        ):
            roots.append(mid)
        else:
            children.setdefault(pid, []).append(mid)

    print(
        f"  Found {len(roots)} English "
        f"conversation trees..."
    )

    def _longest_path(node_id):
        """DFS to find the longest path from
        node_id to a leaf."""
        kids = children.get(node_id, [])
        if not kids:
            return [node_id]
        best = []
        for kid in kids:
            path = _longest_path(kid)
            if len(path) > len(best):
                best = path
        return [node_id] + best

    conversations = []
    for root_id in roots:
        if len(conversations) >= max_conversations:
            break
        path = _longest_path(root_id)
        if len(path) < 2:
            continue

        turns = []
        for nid in path:
            msg = msg_map.get(nid)
            if msg is None:
                continue
            role = _normalize_role(
                msg.get("role", "prompter")
            )
            content = _sanitize_content(
                msg.get("text", "")
            )
            if content:
                turns.append(
                    {"role": role,
                     "content": content}
                )
        if len(turns) < 2:
            continue

        tree_id = str(
            msg_map[root_id].get(
                "message_tree_id", root_id
            )
        )
        conversations.append(
            {"id": tree_id, "turns": turns}
        )
        if (
            len(conversations) % 1000 == 0
            and len(conversations) > 0
        ):
            print(
                f"  Converted "
                f"{len(conversations)} "
                f"conversations..."
            )

    return conversations


def download_and_convert(
    dataset_name: str,
    output_path: str,
    max_conversations: int,
):
    """Download a dataset and write in our schema."""
    if dataset_name not in DATASET_CONFIGS:
        print(
            f"Error: unknown dataset "
            f"'{dataset_name}'. "
            f"Choose from: "
            f"{list(DATASET_CONFIGS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = DATASET_CONFIGS[dataset_name]
    if cfg.get("note"):
        print(f"Note: {cfg['note']}")

    print(
        f"Downloading {dataset_name} from "
        f"{cfg['repo']}..."
    )

    if dataset_name in ("sharegpt", "vicuna"):
        conversations = _load_sharegpt(
            cfg, max_conversations
        )
    elif dataset_name == "lmsys":
        conversations = _load_parquet_dataset(
            cfg, max_conversations,
            _convert_lmsys_row,
        )
    elif dataset_name == "wildchat":
        conversations = _load_parquet_dataset(
            cfg, max_conversations,
            _convert_wildchat_row,
        )
    elif dataset_name == "longbench":
        conversations = _load_longbench(
            cfg, max_conversations
        )
    elif dataset_name == "ultrachat":
        conversations = _load_jsonl_dataset(
            cfg, max_conversations,
            _convert_ultrachat_row,
        )
    elif dataset_name == "oasst":
        conversations = _convert_oasst_dataset(
            cfg, max_conversations
        )
    else:
        conversations = []

    output = {
        "version": "1.0",
        "source": dataset_name,
        "conversations": conversations,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(
        out_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            output, f, indent=2, ensure_ascii=False
        )

    print(
        f"Wrote {len(conversations)} conversations "
        f"to {out_path}"
    )


def reprocess_file(input_path: str):
    """Re-sanitize an existing conversation file."""
    path = Path(input_path)
    if not path.exists():
        print(
            f"Error: file not found: {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    convs = data.get("conversations", [])
    cleaned = 0
    dropped = 0
    for conv in convs:
        new_turns = []
        for turn in conv.get("turns", []):
            content = _sanitize_content(
                turn.get("content", "")
            )
            if content:
                turn["content"] = content
                new_turns.append(turn)
            else:
                dropped += 1
                cleaned += 1
        if new_turns != conv.get("turns", []):
            cleaned += 1
        conv["turns"] = new_turns

    data["conversations"] = [
        c for c in convs
        if len(c.get("turns", [])) >= 2
    ]
    removed = len(convs) - len(data["conversations"])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2, ensure_ascii=False
        )

    print(
        f"Sanitized {path}: "
        f"{len(data['conversations'])} conversations "
        f"kept, {removed} removed "
        f"(< 2 turns after cleaning), "
        f"{dropped} empty turns dropped"
    )
