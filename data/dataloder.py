import hashlib
import json
import pickle
from pathlib import Path

from loguru import logger

from data.load.load_deap import load_deap
from data.load.load_dreamer import load_dreamer
from data.load.load_seed import load_seed
from data.load.load_seed_raw import load_seed_raw
from data.load.load_seed_iv import load_seed_iv
from data.merge_and_split import merge_and_split_deap, merge_and_split_seed
from data.preprocess.preprocess_deap import preprocess_deap
from data.preprocess.preprocess_seed import preprocess_seed
import awkward as ak


_CACHE_VERSION = "v2"


def _to_plain_str(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _build_cache_path(
    cache_dir: str,
    dataset_name: str,
    dataset_path: str,
    sample_length: int = 1,
    stride: int | None = None,
    trim_trial_start_pct: float = 0.0,
    label_type: str | None = None,
) -> Path:
    cache_key_payload = {
        "cache_version": _CACHE_VERSION,
        "dataset_name": dataset_name,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "sample_length": sample_length,
        "stride": stride if stride is not None else sample_length,
        "trim_trial_start_pct": trim_trial_start_pct,
    }
    if label_type is not None:
        cache_key_payload["label_type"] = label_type
    cache_key = hashlib.md5(
        json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    file_name = f"{dataset_name.lower()}_{cache_key}.pkl"
    return Path(cache_dir).expanduser() / file_name


def load_data(
    dataset_name: str,
    dataset_path: str,
    cache_dir: str | None = None,
    sample_length: int = 1,
    stride: int | None = None,
    trim_trial_start_pct: float = 0.0,
    label_type: str = "valence",
) -> tuple[ak.Array, ak.Array, int, int, int, int]:
    """
    return:
        data:  list, shape (session, subject, trail, sample, electrode, feature)
        label: list, shape (session, subject, trail, sample)
        num_subjects
        num_electrodes
        num_features
        num_classes
    """
    logger.info(f"Loading dataset {dataset_name} from path {dataset_path}")

    function_map = {
        "DEAP": load_deap,
        "SEED": load_seed,
        "SEED_IV": load_seed_iv,
        "DREAMER": load_dreamer,
    }

    if _to_plain_str(dataset_name) not in function_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    cache_path = None
    if cache_dir:
        cache_path = _build_cache_path(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            sample_length=sample_length,
            stride=stride,
            trim_trial_start_pct=trim_trial_start_pct,
            label_type=label_type if _to_plain_str(dataset_name) in ("DREAMER", "DEAP") else None,
        )
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cache_payload = pickle.load(f)
                if "result" in cache_payload:
                    logger.info("Dataset cache hit: {}", cache_path)
                    return cache_payload["result"]
                logger.warning("Dataset cache file is invalid, recomputing: {}", cache_path)
            except Exception as exc:
                logger.warning("Failed to load dataset cache {} ({})", cache_path, exc)

    # Load the data and labels
    name = _to_plain_str(dataset_name)
    if name == "SEED" and sample_length > 1:
        result = load_seed_raw(dataset_path, sample_length=sample_length, stride=stride)
    elif name in ("DREAMER", "DEAP"):
        result = function_map[name](dataset_path, label_type=label_type, trim_trial_start_pct=trim_trial_start_pct)
    else:
        result = function_map[name](dataset_path, trim_trial_start_pct=trim_trial_start_pct)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump({"result": result}, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Dataset cache saved: {}", cache_path)
        except Exception as exc:
            logger.warning("Failed to save dataset cache {} ({})", cache_path, exc)

    return result
