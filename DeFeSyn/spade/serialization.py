import base64
import gzip
import hashlib
import io
from typing import Dict, Any

import torch


def _pack(sd: Dict[str, Any]) -> bytes:
    with io.BytesIO() as buf:
        torch.save(sd, buf)           # modern ZIP serialization
        raw = buf.getvalue()
    return gzip.compress(raw)

def _unpack(blob: bytes) -> Dict[str, Any]:
    buf = io.BytesIO(gzip.decompress(blob))
    obj = torch.load(buf, map_location="cpu")
    # Normalize dtypes (float64 -> float32) for cross-env consistency
    for k, v in list(obj.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            obj[k] = v.float()
    return obj


def encode_state_dict_pair(gen_sd: Dict[str, Any], dis_sd: Dict[str, Any]) -> Dict[str, Any]:
    gen_comp = _pack(gen_sd)
    dis_comp = _pack(dis_sd)
    checksum = hashlib.sha256(gen_comp + dis_comp).hexdigest()
    return {
        "generator": base64.b64encode(gen_comp).decode("utf-8"),
        "discriminator": base64.b64encode(dis_comp).decode("utf-8"),
        "checksum": checksum,
        "gen_bytes": len(gen_comp),
        "dis_bytes": len(dis_comp),
    }

def decode_state_dict_pair(package: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    gen_comp = base64.b64decode(package["generator"])
    dis_comp = base64.b64decode(package["discriminator"])
    checksum = package["checksum"]
    if hashlib.sha256(gen_comp + dis_comp).hexdigest() != checksum:
        raise ValueError("Checksum mismatch.")
    gen_sd = _unpack(gen_comp)
    dis_sd = _unpack(dis_comp)
    return gen_sd, dis_sd